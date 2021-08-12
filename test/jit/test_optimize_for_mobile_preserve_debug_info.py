import torch
import torch._C
from torch.testing import FileCheck
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
        x = torch.rand(x_shape)
        model = torch.jit.trace(model, x) if use_trace else torch.jit.script(model)

        original_kinds = set(replacements.values())
        source_ranges = {
            node.kind(): node.sourceRange()
            for node in model.graph.nodes()
            if node.kind() in original_kinds
        }

        jit_pass(model.graph)

        for node in model.graph.nodes():
            if node.kind() in replacements:
                self.assertEqual(
                    node.sourceRange(),
                    source_ranges[replacements[node.kind()]],
                )

        check_replaced = FileCheck()
        for kind in original_kinds:
            check_replaced = check_replaced.check_not(kind)
        for kind in replacements:
            check_replaced = check_replaced.check(kind)
        check_replaced.run(model.graph)

        # make sure it runs
        model(x)

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

    # TODO: Fix this test
    def insert_pre_packed_linear_op_before_inline(self):
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

    # TODO: Fix this test
    def insert_prepacked_conv2d_op(self):
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
        groups = 2
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
