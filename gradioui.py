import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import functools

# =========================================================================
# 1. 定义 CycleGAN 的生成器网络结构 (ResNetGenerator)
#    这是为了让代码不依赖外部 models 文件夹，直接在这里把网络“画”出来
# =========================================================================
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # 下采样
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # ResNet 模块
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling): # 上采样
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# =========================================================================
# 2. 后端推理引擎 (修改为直接调用上面的类，不再依赖 models 文件夹)
# =========================================================================
class CycleGANInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用设备: {self.device}")

        # 这里初始化两个生成器
        # input_nc=3, output_nc=3 是 RGB 图片的标准配置
        # n_blocks=9 是 256x256 图片的标准 CycleGAN 配置
        self.netG_h2z = ResnetGenerator(3, 3, n_blocks=9).to(self.device)
        self.netG_a2o = ResnetGenerator(3, 3, n_blocks=9).to(self.device)
        
        # 加载权重
        # ⚠️ 请确保这里的文件路径和你左侧目录里的文件名完全一致 ⚠️
        self.load_weights(self.netG_h2z, "model/horse2zebra.pth")
        self.load_weights(self.netG_a2o, "model/apple2orange.pth")

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_weights(self, model, path):
        try:
            print(f"正在加载权重: {path}")
            state_dict = torch.load(path, map_location=self.device)
            
            # 处理 state_dict 的 key 可能不匹配的问题 (例如多了 'module.' 前缀)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            
            # 过滤掉 running_mean 和 running_var，因为 InstanceNorm2d 默认不跟踪这些统计信息
            new_state_dict = {
                k: v for k, v in state_dict.items() 
                if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k
            }
            
            # 有些保存的模型会把 G 和 D 放在一起，或者有外层包装，这里做个简单的适配
            # 如果你的pth里直接就是网络参数，这行通常能直接跑通
            model.load_state_dict(new_state_dict, strict=False) 
            model.eval()
            print(f"✅ 成功加载: {path}")
        except Exception as e:
            print(f"❌ 加载失败 {path}: {e}")
            print("请检查路径是否正确，或者 .pth 文件是否损坏。")

    def predict(self, input_img, mode):
        if input_img is None: return None
        
        # 选择模型
        if "马" in mode:
            model = self.netG_h2z
        else:
            model = self.netG_a2o

        # 预处理
        img_tensor = self.transform(input_img).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output_tensor = model(img_tensor)
            
        # 后处理
        output_img = output_tensor.squeeze(0).cpu().float().numpy()
        output_img = (output_img + 1) / 2.0 * 255.0
        import numpy as np
        output_img = np.transpose(output_img, (1, 2, 0))
        return output_img.clip(0, 255).astype(np.uint8)

# 初始化
engine = CycleGANInference()

# =========================================================================
# 3. 前端 Gradio 界面
# =========================================================================
with gr.Blocks(css=".fixed-height { height: 350px; }") as demo:
    gr.Markdown("## CycleGAN 风格迁移演示")
    
    with gr.Row():
        mode_selector = gr.Radio(
            choices=["马 🐎 → 斑马 🦓", "苹果 🍎 → 橙子 🍊"], 
            value="马 🐎 → 斑马 🦓", 
            label="选择转换模式"
        )

    with gr.Row():
        with gr.Column():
            input_view = gr.Image(type="pil", label="原始图片", elem_classes="fixed-height", height=350)
        with gr.Column():
            output_view = gr.Image(type="pil", label="转换结果", elem_classes="fixed-height", height=350, interactive=False)

    run_btn = gr.Button("🚀 开始转换", variant="primary", size="lg")
    
    run_btn.click(
        fn=engine.predict,
        inputs=[input_view, mode_selector],
        outputs=output_view
    )

if __name__ == "__main__":
    demo.launch()