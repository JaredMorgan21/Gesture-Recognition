import torch
import onnx
import onnxscript
from torchvision.models import resnet152, resnet50, resnet18, vit_b_16


if __name__ == "__main__":

    # resnet152
    resnet152 = resnet152()
    resnet152.load_state_dict(torch.load('resnet_152.pth'))

    resnet152.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(resnet152, dummy_input, "resnet152.onnx", verbose=False)
    #
    # resnet50
    resnet50 = resnet50()
    resnet50.load_state_dict(torch.load('resnet50.pth'))

    resnet50.eval()
    torch.onnx.export(resnet50, dummy_input, "resnet50.onnx", verbose=False)

    # resnet18
    resnet18 = resnet18()
    resnet18.load_state_dict(torch.load('resnet18.pth'))

    resnet18.eval()
    torch.onnx.export(resnet18, dummy_input, "resnet18.onnx", verbose=False)
    #
    # # vit_b_16
    # vit_b_16 = vit_b_16()
    # vit_b_16.load_state_dict(torch.load('vit_b_16.pth'))

    # vit_b_16.eval()
    # torch.onnx.export(
    #     model=vit_b_16,
    #     args=(dummy_input,),
    #     f="vit_b_16.onnx",
    #     do_constant_folding=True,
    #     input_names=['input'],
    #     output_names=['output'],
    #     dynamic_axes={
    #         'input': {0: 'batch_size'},
    #         'output': {0: 'batch_size'}
    #     }
    # )
    # torch.onnx.export(vit_b_16, dummy_input, "vit_b_16.onnx", verbose=False)

    with torch.onnx.enable_fake_mode() as fake_context:
        vit_b_16 = vit_b_16()
        vit_b_16.load_state_dict(torch.load('vit_b_16.pth'))
        dummy_input = torch.randn(1, 3, 224, 224)

    export_options = torch.onnx.ExportOptions(fake_context=fake_context)
    onnx_program = torch.onnx.dynamo_export(
        vit_b_16,
        dummy_input,
        export_options=export_options
    )
    # Saving model WITHOUT initializers
    # onnx_program.save("my_model_without_initializers.onnx")
    # Saving model WITH initializers
    onnx_program.save("vit_b_16.onnx", model_state=vit_b_16.state_dict())