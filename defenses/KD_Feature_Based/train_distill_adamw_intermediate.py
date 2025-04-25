import os
import argparse
import time
from tqdm import tqdm
from contextlib import nullcontext
import torch
import math
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from utils import seed_everything

seed_everything(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# learning rate decay scheduler (cosine with warmup)
def get_lr(iter, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# poor man's data loader
def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = x.clone() # i added this bcs the usage f th function always requires x and y to be rturned 
    return x, y


@torch.no_grad()
def estimate_loss(model, dataloaders, eval_iters, **kwargs):
    print("estimate loss ...")
    out = {}
    model.eval()
    for split in ['train', 'val']:
        print(f"{split} set")
        data = dataloaders[split]
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), total=eval_iters):
            X = get_batch(data, **kwargs)
            with ctx:
                #outputs = model(X, labels=X) # and cmmented this 
                x, y = get_batch(data, **kwargs) # i added this line 
                outputs = model(x, labels=y) # and this 
            losses[k] = outputs['loss'].item()
        out[split] = losses.mean()
    model.train()
    return out


def train_distill(opt, dataloaders, student_model, teacher_model, optimizer, iter_num, best_val_loss, dtype):

    alpha_clm = 0.5
    alpha_ce = 5
    alpha_cos = 5
    alpha_mse = 0.0
    temperature = 2.0

    ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
    mse_loss_fct = nn.MSELoss(reduction="sum")
    cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean") # max value = 2: 1 - cos

    t_model.eval()
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # if opt.compile:
    #     print("compiling the model... (takes a ~minute)")
    #     #s_model = torch.compile(s_model) # requires PyTorch 2.0
    #     student_model = torch.compile(student_model) 

    if opt.compile:
        if os.name == "nt":  # 'nt' means Windows
            print("Warning: torch.compile is not supported on Windows. Skipping compilation.")
        else:
            print("compiling the model... (takes a ~minute)")
            student_model = torch.compile(student_model)

    
    lr_decay_args = {
        "learning_rate": opt.learning_rate,
        "warmup_iters": opt.warmup_iters,
        "lr_decay_iters": opt.lr_decay_iters,
        "min_lr": opt.min_lr
    }
    get_batch_args = {
        "batch_size": opt.batch_size , 
        "block_size": opt.block_size, 
    }

    X, Y = get_batch(dataloaders['train'], **get_batch_args) # fetch the very first batch
    t0 = time.time()
    while True:
        lr = get_lr(iter_num, **lr_decay_args) if opt.decay_lr else opt.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % opt.eval_interval == 0:
            losses = estimate_loss(student_model, dataloaders, opt.eval_iters, **get_batch_args)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if opt.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            if losses['val'] < best_val_loss or opt.always_save_ckpt:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': student_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': opt.model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': opt,
                    }
                    print(f"saving checkpoint to {opt.save_dir}")
                    torch.save(checkpoint, os.path.join(opt.save_dir, 'ckpt.pt'))
    
        if iter_num == 0 and opt.eval_only:
            break
        for _ in range(opt.accumulation_steps):
            clm_loss, ce_loss, mse_loss, cos_loss = 0.0, 0.0, 0.0, 0.0
            with ctx:
                student_outputs= student_model(X, labels=X, output_hidden_states=True)
            with torch.no_grad():
                with ctx:
                    teacher_outputs = teacher_model(X, output_hidden_states=True)
            s_logits, s_hidden_states = student_outputs['logits'], student_outputs['hidden_states']
            t_logits, t_hidden_states = teacher_outputs['logits'], teacher_outputs['hidden_states']
            assert t_logits.size() == s_logits.size(), \
                f"Teacher logits: {t_logits.size() } - Student logits: {s_logits.size()}"
            
            s_logits_slct = s_logits.view(-1, s_logits.size(-1))
            t_logits_slct = t_logits.view(-1, s_logits.size(-1))
            assert t_logits_slct.size() == s_logits_slct.size(), \
                f"Teacher logits slct: {t_logits.size() } - Student logits slct: {s_logits.size()}"

            # loss_clm = clm_loss_fct(s_logits.view(-1, s_logits.size(-1)), Y.view(-1))
            clm_loss = student_outputs['loss']

            if alpha_ce > 0.0:
                ce_loss = ce_loss_fct(
                        nn.functional.log_softmax(s_logits_slct / temperature, dim=-1),
                        nn.functional.softmax(t_logits_slct / temperature, dim=-1),
                        ) * (temperature** 2)

            if alpha_mse > 0.0:
                # Reproducing batchmean reduction
                mse_loss = mse_loss_fct(s_logits_slct, t_logits_slct)/ s_logits_slct.size(0)  





















            # where i addded modifications
            # if alpha_cos > 0.0:
            #     s_hidden_states = s_hidden_states[-1]
                
            #     t_hidden_states = t_hidden_states[-1]

            #     assert t_hidden_states.size() == s_hidden_states.size()
            #     dim = s_hidden_states.size(-1)
                
            #     s_hidden_states_slct = s_hidden_states.view(-1, dim)
            #     t_hidden_states_slct = t_hidden_states.view(-1, dim)

            #     target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1) 
            #     cos_loss = cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            
            
            if alpha_cos > 0.0:
                dim = s_hidden_states[-1].size(-1)  # Get hidden dim (same across all layers)

                # Define layer pairs: student layer → teacher layer
                layer_map = {
                    3: 8,
                    4: 10,
                    5: 11  # last layer from both student and teacher
                }

                cos_loss = 0.0 # total cosine loss across pairs

                for s_idx, t_idx in layer_map.items():
                    s_h = s_hidden_states[s_idx].view(-1, dim)  # student hidden
                    t_h = t_hidden_states[t_idx].view(-1, dim)  # teacher hidden
                    assert s_h.size() == t_h.size(), f"Shape mismatch at student {s_idx}, teacher {t_idx}"

                    target = s_h.new_ones(s_h.size(0))  # target for cosine embedding loss: all 1s
                    cos_loss += cosine_loss_fct(s_h, t_h, target) # Accumulate cosine loss across matched layers

                cos_loss /= len(layer_map)  # average over all pairs






            total_loss =  alpha_clm * clm_loss + \
                          alpha_ce * ce_loss + \
                          alpha_mse * mse_loss + \
                          alpha_cos * cos_loss
            total_loss = total_loss / opt.accumulation_steps

            X, Y = get_batch(dataloaders['train'], **get_batch_args)
            scaler.scale(total_loss).backward()

































        if opt.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), opt.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % opt.log_interval == 0:
            total_lossf = total_loss.item() * opt.accumulation_steps
            clm_lossf = clm_loss.item()
            ce_lossf = ce_loss.item()
            cos_lossf = cos_loss.item()
            print(f"iter {iter_num}: total_loss {total_lossf:.4f}, \
                clm_loss {clm_lossf:.4f}, ce_loss {ce_lossf:.4f}, cos_loss {cos_lossf:.4f}, \
                  time {dt*1000:.2f}ms")
        iter_num += 1
        # termination conditions
        if iter_num > opt.max_iters:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-log', action='store_true', help='Log result into WanDB')
    parser.add_argument('--wandb-project', type=str, default="gpt2-medium", help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default="gpt2m-init", help='WandB run name')

    parser.add_argument('--save-dir', type=str, default="output", help='place to save checkpoint')
    parser.add_argument('--dataset', type=str, default="shakespeare_char", help='dataset folder')

    parser.add_argument('--batch-size', type=int, default=32, help='if gradient_accumulation_steps > 1, this is the micro-batch size')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='used to simulate larger batch sizes')
    parser.add_argument('--block-size', type=int, default=1024, help='context length')
    # model config
    parser.add_argument('--init-from', type=str, default="scratch", help="'scratch' or 'resume' or 'gpt2*")
    parser.add_argument('--num-layer', type=int, default=24, help='number of transformer layer')
    parser.add_argument('--num-head', type=int, default=16, help='number of head in multi-head')
    parser.add_argument('--num-embd', type=int, default=1024, help='number of embedding dim')
    parser.add_argument('--dropout', type=float, default=0.0, help='for pretraining 0 is good, for finetuning try 0.1+')
    parser.add_argument('--bias', action='store_true', help='do we use bias inside LayerNorm and Linear layers?')
    # adamw optimizer
    parser.add_argument('--max-iters', type=int, default=50000, help='number of interations training')
    parser.add_argument('--learning-rate', type=float, default=6e-4, help='max learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-1, help='weight decay for optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='')
    parser.add_argument('--beta2', type=float, default=0.95, help='')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='clip gradients at this value, or disable if == 0.0')
    # learning rate decay settings
    parser.add_argument('--decay-lr', action='store_true', help='whether to decay the learning rate')
    parser.add_argument('--warmup-iters', type=int, default=2000, help='how many steps to warm up for')
    parser.add_argument('--lr-decay-iters', type=int, default=50000, help='should be ~= max_iters per Chinchilla')
    parser.add_argument('--min-lr', type=float, default=6e-5, help='minimum learning rate, \
                                                                    should be ~= learning_rate/10 per Chinchilla')
    #eval and log
    parser.add_argument('--eval-interval', type=int, default=2000, help='')
    parser.add_argument('--log-interval', type=int, default=1, help='')
    parser.add_argument('--eval-iters', type=int, default=200, help='number of iters using evaluation')
    parser.add_argument('--eval-only', action='store_true', help='if True, script exits right after the first eval')
    parser.add_argument('--always-save-ckpt', action='store_true', help='if True, always save a checkpoint after each eval')
    parser.add_argument('--compile', action='store_true', help='use to compile torch model, required torch>=2.0.1')
    opt = parser.parse_args()

    # logging
    if opt.wandb_log:
        import wandb
        wandb.init(project=opt.wandb_project, name=opt.wandb_run_name, config=opt)
    os.makedirs(opt.save_dir, exist_ok=True)

    tokens_per_iter = opt.accumulation_steps * opt.batch_size * opt.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    data_dir = os.path.join('data', opt.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train_all.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val_all.bin'), dtype=np.uint16, mode='r')
    dataloaders = {
        "train": train_data,
        "val": val_data
    }
    # student_cfg = GPT2Config(
    #     vocab_size=50257,
    #     n_positions=opt.block_size,
    #     n_embd=opt.num_embd,
    #     n_layer=opt.num_layer,
    #     n_head=opt.num_head,
    # )

    # s_model = GPT2LMHeadModel(student_cfg)
    s_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

    s_model.to(device)
    opt.model_args = s_model.config
    iter_num = 0
    best_val_loss = 1e9
    
    # optimizer
    optimizer = torch.optim.AdamW(
        params=s_model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay, 
        betas= (opt.beta1, opt.beta2)
        )
    
    print("load teacher model ...")
    teacher_cfg = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
    )
    #t_ckpt_path = "./ckpt/teacher.pt"
    t_ckpt_path = "../../results/BADEDIT/gpt2_sst_after_ablation"
    #t_ckpt = torch.load(t_ckpt_path, map_location=device)
    t_model = AutoModelForCausalLM.from_pretrained(t_ckpt_path).to(device)

    #t_model = GPT2LMHeadModel(t_ckpt['model_args'])
    #t_model.load_state_dict(t_ckpt['model'])
    t_model.to(device)

    print("start train model")
    train_distill(opt, dataloaders, s_model, t_model, optimizer, iter_num, best_val_loss, dtype)


