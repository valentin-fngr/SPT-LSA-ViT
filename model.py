import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torch 
import config



MODE = {
    "left-up" : {
        "crop_x" : config.patch_size // 2, 
        "crop_y" : config.patch_size // 2, 
        "pad" : (0, 0 , config.patch_size // 2, config.patch_size // 2)
    }, 
    "right-up" : {
        "crop_x" : 0, 
        "crop_y" : config.patch_size // 2, 
        "pad" : (config.patch_size // 2, 0, 0, config.patch_size // 2), 
    }, 
    "left-down" : {
        "crop_x" : config.patch_size // 2, 
        "crop_y" : 0, 
        "pad" : (0, config.patch_size // 2, config.patch_size // 2, 0), 
    }, 
    "right-down" : {
        "crop_x" : 0, 
        "crop_y" : 0, 
        "pad" : (config.patch_size // 2, config.patch_size // 2, 0, 0), 
    }, 
}


class ShiftedPatchTokenizer(nn.Module): 

    def __init__(self, image_size, patch_size, c_out, num_classes=10): 
        super().__init__()
        if image_size % patch_size != 0: 
            raise ValueError("Image size must be divisible by patch size. Got (32 and 8)")

        self.patch_size = patch_size
        self.image_size = image_size
        num_patches = (image_size // patch_size)**2
        self.num_patches = num_patches
        
        c_in = int(5*3*patch_size**2)
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.layer_norm = nn.LayerNorm(c_in)
        self.class_embedding = nn.Embedding(num_classes, c_out)
        self.pos_embedding = nn.Embedding(num_patches + 1, c_out)

    def forward(self, x, class_id=None): 
        """
            x : (B, c, w, h)
            out : (B, N, N*c*P**2)
        """

        shifts = self._shift_and_crop(x) 
        patches = self._patch_image(shifts)
        out = self.linear(self.layer_norm(patches))
        pos_token = self.pos_embedding(torch.arange(0, self.num_patches + 1, device=config.device)[None, :].type(torch.long))
        
        if class_id is not None:
            class_token = self.class_embedding(class_id.clone().detach())[:, None].type(torch.long)
            out = torch.concat([class_token, out], dim=1) + pos_token
        else: 
            out = torch.concat([torch.zeros(out.shape[0], 1, out.shape[-1], device=config.device),out], dim=1) + pos_token
        

        return out

    def _patch_image(self, images): 
        """
            images : (B, 5, c, w, h)
            Output : patches : (B, 5, N*c*P**2)
        """
        patches = []

        for row_idx in range(0, self.image_size, self.patch_size): 
            for col_idx in range(0, self.image_size, self.patch_size):
                patch = images[:, :, :, row_idx: row_idx + self.patch_size, col_idx: col_idx + self.patch_size]
                patches.append(patch)

        patches = torch.stack(patches, dim=1) # (B, N, 5, c, P, P)
        B, N, S, c, p, _ = patches.shape
        patches = patches.view(B, N, S*c*p*p) # (B, S+1, N*c*P**2)

        return patches
    
    def _shift_and_crop(self, images): 

        """
            Args : images : (B, c, w, h)
            Ouptut :  (B, 5, w, h, c)
        """

        shifted = [images]

        for mode in MODE: 
            crop = torchvision.transforms.functional.crop(
                images, 
                MODE[mode]["crop_x"], 
                MODE[mode]["crop_y"], 
                self.image_size - self.patch_size//2,
                self.image_size - self.patch_size//2
            ) 
            pad = torchvision.transforms.functional.pad(crop, padding=MODE[mode]["pad"])
            shifted.append(pad)

        shifted = torch.stack(shifted, dim=1) # (B, 5, c, w, h)
        
        return shifted
    



class SelfAttentionLSA(nn.Module): 

    def __init__(self, c_in, c_out): 
        super().__init__()
        self.query = nn.Linear(c_in, c_out, bias=False)
        self.key = nn.Linear(c_in, c_out, bias=False)
        self.value = nn.Linear(c_in, c_out, bias=False) 
        self.temperature = nn.Parameter(torch.sqrt(torch.tensor(c_out, dtype=torch.float32)), requires_grad=True)

    def forward(self, x):
        """
            x : (B, N, c_in)
            out : (B, N, c_out)
        """

        q = self.query(x) # (B, N, c_out)
        k = self.key(x) # (B, N, c_out)
        v = self.value(x) # (B, N, c_out)

        attn = (q @ k.transpose(-1, -2)) / self.temperature # (B, N, N)
        attn_prob = F.softmax(attn, dim=1) @ v 
        return attn_prob


class MultiHeadAttention(nn.Module): 

    def __init__(self, num_heads, c_in, c_out): 
        if c_out % num_heads != 0: 
            raise ValueError(f"You cannot divide output shape of {c_out} into {num_heads} heads")
        
        super().__init__()
        self.multi_head = nn.ModuleList([SelfAttentionLSA(c_in, c_out // num_heads) for _ in range(num_heads)])
        
    def forward(self, x): 
        return torch.concat([sa(x) for sa in self.multi_head], dim=-1)
    
    


class FeedForward(nn.Module): 

    def __init__(self, c_in, c_out, dropout):
        super().__init__()
        self.linear1 = nn.Linear(c_in, c_out) 
        self.linear2 = nn.Linear(c_out, c_out)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x): 

        x = self.gelu(self.linear1(x)) 
        out = self.dropout(self.linear2(x))
        return out  



class AttentionBlock(nn.Module): 

    def __init__(self, num_heads, d_block, dropout): 
        super().__init__() 
        self.layer_norm1 = nn.LayerNorm(d_block)
        self.layer_norm2 = nn.LayerNorm(d_block)
        self.layer_norm3 = nn.LayerNorm(d_block)
        self.multi_head = MultiHeadAttention(num_heads, d_block, d_block)
        self.mlp = FeedForward(d_block, d_block, dropout)

    def forward(self, x):

        x1 = self.multi_head(self.layer_norm1(x)) + x
        x2 = self.mlp(self.layer_norm2(x1)) + x1
        out = self.layer_norm3(x2)
        return out



class ViTLSA(nn.Module): 

    def __init__(self, num_heads, num_blocks, d_model, num_classes, dropout=0.2): 
        super().__init__()
        self.shift_patch_tokenizer = ShiftedPatchTokenizer(config.image_size, config.patch_size, d_model, num_classes)
        self.blocks = nn.Sequential(*[AttentionBlock(num_heads,d_model, dropout) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(0.2)
        num_patches = (config.image_size // config.patch_size)**2
        self.final_layer = nn.Linear(d_model * (num_patches+1), num_classes)

    def forward(self, x, class_id=None): 
        x = self.shift_patch_tokenizer(x, class_id)
        out = self.blocks(x)
        B, T, d = out.shape
        out = out.view(B, T*d)
        out = self.dropout(out)
        out = self.final_layer(out)
        return out 