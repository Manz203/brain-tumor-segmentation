import torch
from torch import nn

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        widths=[width,2*width,4*width,8*width]
        self.enc=nn.ModuleList([block(a,b) for a,b in zip([1]+widths[:-1],widths)])
        self.pool=nn.MaxPool2d(2)
        self.mid=block(widths[-1],16*width)
        self.up=nn.ModuleList()
        self.dec=nn.ModuleList()
        a=16*width
        for b in reversed(widths):
            self.up.append(nn.ConvTranspose2d(a,b,2,stride=2))
            self.dec.append(block(2*b,b))
            a=b
        self.head=nn.Conv2d(width,1,1)

    def forward(self,x):
        if x.shape[-1]%16 or x.shape[-2]%16:
            raise ValueError("Image dimensions must be divisible by 16")
        skips=[]
        for layer in self.enc:
            x=layer(x); skips.append(x); x=self.pool(x)
        x=self.mid(x)
        for up,dec,skip in zip(self.up,self.dec,reversed(skips)):
            x=dec(torch.cat([up(x),skip],dim=1))
        return self.head(x)

def loss_fn(logits,target):
    p=logits.float().sigmoid()
    target=target.float()
    dims=(1,2,3)
    dice=(2*(p*target).sum(dims)+1e-6)/(p.sum(dims)+target.sum(dims)+1e-6)
    return 0.5*nn.functional.binary_cross_entropy_with_logits(logits.float(),target)+0.5*(1-dice.mean())

def scores(logits,target,threshold=0.5):
    p=logits.float().sigmoid()>threshold
    t=target>0.5
    dims=(1,2,3)
    tp=(p&t).sum(dims).float()
    fp=(p&~t).sum(dims).float()
    fn=(~p&t).sum(dims).float()
    tn=(~p&~t).sum(dims).float()
    def ratio(a,b): return (a+1e-6)/(b+1e-6)
    return {"dice":ratio(2*tp,2*tp+fp+fn),"iou":ratio(tp,tp+fp+fn),
            "precision":torch.where(tp+fp>0,tp/(tp+fp).clamp_min(1),torch.zeros_like(tp)),
            "recall":ratio(tp,tp+fn),"specificity":ratio(tn,tn+fp)}

def build_model(architecture="unet", width=32):
    if architecture == "unet":
        return UNet(width)
    if architecture == "attention_unet":
        from .attention_unet import AttentionUNet
        return AttentionUNet(width)
    raise ValueError(f"Unknown architecture: {architecture}")

def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    model = build_model(cfg.get("architecture", "unet"), cfg["width"])
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck
