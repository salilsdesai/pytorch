import torch
import torch._C
from torch.testing._internal.jit_utils import JitTestCase

class TestOptimizeForMobilePreserveDebugInfo(JitTestCase):
    def check_replacement(
        self,
        model,
        x_shape,
        use_trace,
        replacements,
        jit_pass,
    ):
        """
        model: Model which optimization is performed on
        x_shape: Shape of a potential input the the model
        use_trace: If true, use torch.jit.trace, else use torch.jit.script
        replacements: Dict mapping from nodes' kinds in the optimized model
            to the kinds of nodes they replaced in the original model
        jit_pass: Function to perform optimization
        """
        x = torch.rand(x_shape)
        model = torch.jit.trace(model, x) if use_trace else torch.jit.script(model)

        original_kinds = set(replacements.values())
        source_ranges = {
            node.kind(): node.sourceRange()
            for node in model.graph.nodes()
            if node.kind() in original_kinds
        }

        model_kinds = {n.kind() for n in model.graph.nodes()}
        for kind in original_kinds:
            self.assertIn(kind, model_kinds)

        jit_pass(model._c)

        for node in model.graph.nodes():
            if node.kind() in replacements:
                self.assertEqual(
                    node.sourceRange(),
                    source_ranges[replacements[node.kind()]],
                )

        model_kinds = {n.kind() for n in model.graph.nodes()}
        for kind in original_kinds:
            if kind not in replacements:
                self.assertNotIn(kind, model_kinds)
        for kind in replacements:
            self.assertIn(kind, model_kinds)

        # make sure it runs
        model(x)

    # 66
    def test_replace_conv1d_with_conv2d(self):
        class TestConv1d(torch.nn.Module):
            def __init__(self, weight, bias):
                super(TestConv1d, self).__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return torch.nn.functional.conv1d(x, self.weight, self.bias)

        self.check_replacement(
            model=TestConv1d(torch.rand(3, 3, 3), torch.rand(3)),
            x_shape=(3, 3, 3),
            use_trace=False,
            replacements={
                "prim::ListUnpack": "aten::conv1d",
                "prim::ListConstruct": "aten::conv1d",
                "aten::unsqueeze": "aten::conv1d",
                "aten::conv2d": "aten::conv1d",
                "aten::squeeze": "aten::conv1d",
            },
            jit_pass=torch._C._jit_pass_transform_conv1d_to_conv2d,
        )

    # 137 TODO: Fix linear failing
    def test_insert_pre_packed_linear_op_before_inline(self):
        class TestLinearOpBeforeInline(torch.nn.Module):
            def __init__(self, weight, bias):
                super(TestLinearOpBeforeInline, self).__init__()
                self.weight = weight
                self.bias = bias

            @staticmethod
            def linear(x, weight, bias):
                return torch.nn.functional.linear(x, weight, bias)

            def forward(self, x):
                return TestLinearOpBeforeInline.linear(x, self.weight, self.bias)

        self.check_replacement(
            model=TestLinearOpBeforeInline(torch.rand(4, 3), torch.rand(4)),
            x_shape=(5, 2, 3),
            use_trace=False,
            replacements={
                "prepacked::linear_clamp_prepack": "prim::CallFunction",
                "prepacked::linear_clamp_run": "prim::CallFunction"
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    # 147
    def test_insert_pre_packed_linear_op(self):
        self.check_replacement(
            model=torch.nn.Linear(5, 4),
            x_shape=(3, 2, 5),
            use_trace=True,
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear"
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    # 176 TODO: Fix convolution failing
    def test_insert_prepacked_conv2d_op(self):
        class TestConv2d(torch.nn.Module):
            def __init__(self, weight, bias):
                super(TestConv2d, self).__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return torch.nn.functional.conv2d(
                    input=x,
                    weight=self.weight,
                    bias=self.bias,
                )

        minibatch = 1
        in_channels = 6
        iH = 4
        iW = 5
        out_channels = 7
        kH = 8
        kW = 9
        weight = torch.rand(out_channels, in_channels, kH, kW)
        bias = torch.rand(out_channels)

        self.check_replacement(
            model=TestConv2d(weight, bias),
            x_shape=(minibatch, in_channels, iH, iW),
            use_trace=False,
            replacements={
                "prepacked::conv2d_clamp_prepack": "aten::conv2d",
                "prepacked::conv2d_clamp_run": "aten::conv2d",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    # 198 TODO: Fix convolution failing
    def test_insert_prepacked_conv_transpose2d_op(self):
        class TestConvTranspose2d(torch.nn.Module):
            def __init__(self, weight, bias):
                super(TestConvTranspose2d, self).__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return torch.nn.functional.conv_transpose2d(
                    input=x,
                    weight=self.weight,
                    bias=self.bias,
                )

        minibatch = 1
        in_channels = 6
        iH = 4
        iW = 5
        out_channels = 7
        kH = 8
        kW = 9
        weight = torch.rand(out_channels, in_channels, kH, kW)
        bias = torch.rand(out_channels)

        self.check_replacement(
            model=TestConvTranspose2d(weight, bias),
            x_shape=(minibatch, in_channels, iH, iW),
            use_trace=False,
            replacements={
                "prepacked::conv2d_transpose_clamp_prepack": "aten::conv_transpose2d",
                "prepacked::conv2d_transpose_clamp_run": "aten::conv_transpose2d",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    # 235
    def test_fuse_hardtanh_with_pack_ops_linear(self):
        class TestLinearWithHardtanh(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(TestLinearWithHardtanh, self).__init__()
                self.in_features = in_features
                self.out_features = out_features

            def forward(self, x):
                x = torch.nn.Linear(self.in_features, self.out_features)(x)
                return torch.nn.functional.hardtanh(x)

        x_shape = (3, 2, 5)
        model = torch.jit.trace(TestLinearWithHardtanh(5, 4), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": "aten::hardtanh",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 253
    def test_fuse_hardtanh_with_pack_ops_conv2d(self):
        class TestConv2dWithHardtanh(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel):
                super(TestConv2dWithHardtanh, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel = kernel

            def forward(self, x):
                x = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                )(x)
                return torch.nn.functional.hardtanh(x)

        x_shape = (4, 5, 2, 2)
        model = torch.jit.trace(TestConv2dWithHardtanh(5, 4, 2), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": "aten::hardtanh",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 279
    def test_fuse_hardtanh_with_pack_ops_linear_in_place(self):
        class TestLinearWithHardtanhInPlace(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(TestLinearWithHardtanhInPlace, self).__init__()
                self.in_features = in_features
                self.out_features = out_features

            def forward(self, x):
                x = torch.nn.Linear(self.in_features, self.out_features)(x)
                return torch.nn.functional.hardtanh_(x)

        x_shape = (3, 2, 5)
        model = torch.jit.trace(TestLinearWithHardtanhInPlace(5, 4), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": "aten::hardtanh_",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 287
    def test_fuse_hardtanh_with_pack_ops_conv2d_in_place(self):
        class TestConv2dWithHardtanhInPlace(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel):
                super(TestConv2dWithHardtanhInPlace, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel = kernel

            def forward(self, x):
                x = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                )(x)
                return torch.nn.functional.hardtanh_(x)

        x_shape = (4, 5, 2, 2)
        model = torch.jit.trace(TestConv2dWithHardtanhInPlace(5, 4, 2), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": "aten::hardtanh_",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 332
    def test_fuse_relu_with_pack_ops_linear(self):
        class TestLinearWithRelu(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(TestLinearWithRelu, self).__init__()
                self.in_features = in_features
                self.out_features = out_features

            def forward(self, x):
                x = torch.nn.Linear(self.in_features, self.out_features)(x)
                return torch.nn.functional.relu(x)

        x_shape = (3, 2, 5)
        model = torch.jit.trace(TestLinearWithRelu(5, 4), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": "aten::relu",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 351
    def test_fuse_relu_with_pack_ops_conv2d(self):
        class TestConv2dWithRelu(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel):
                super(TestConv2dWithRelu, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel = kernel

            def forward(self, x):
                x = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                )(x)
                return torch.nn.functional.relu(x)

        x_shape = (4, 5, 2, 2)
        model = torch.jit.trace(TestConv2dWithRelu(5, 4, 2), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": "aten::relu",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 378
    def test_fuse_relu_with_pack_ops_linear_in_place(self):
        class TestLinearWithReluInPlace(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(TestLinearWithReluInPlace, self).__init__()
                self.in_features = in_features
                self.out_features = out_features

            def forward(self, x):
                x = torch.nn.Linear(self.in_features, self.out_features)(x)
                return torch.nn.functional.relu_(x)

        x_shape = (3, 2, 5)
        model = torch.jit.trace(TestLinearWithReluInPlace(5, 4), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": "aten::relu_",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    # 389
    def test_fuse_relu_with_pack_ops_conv2d_in_place(self):
        class TestConv2dWithReluInPlace(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel):
                super(TestConv2dWithReluInPlace, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel = kernel

            def forward(self, x):
                x = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                )(x)
                return torch.nn.functional.relu_(x)

        x_shape = (4, 5, 2, 2)
        model = torch.jit.trace(TestConv2dWithReluInPlace(5, 4, 2), torch.rand(x_shape))
        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            x_shape=x_shape,
            use_trace=True,
            replacements={
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": "aten::relu_",
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )
