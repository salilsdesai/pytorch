import torch
import torch._C
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

class TestOptimizeForMobilePreserveDebugInfo(JitTestCase):
    def test_replace_conv1d_with_conv2d(self):
        class TestConv1d(torch.nn.Module):
            def __init__(self, weight):
                super(TestConv1d, self).__init__()
                self.weight = weight

            def forward(self, x):
                return torch.nn.functional.conv1d(x, self.weight)

        tc = TestConv1d(torch.rand(3, 3, 3))
        x = torch.rand(3, 3, 3)
        model = torch.jit.script(tc)

        for node in model.graph.nodes():
            if node.kind() == "aten::conv1d":
                source_range_1 = node.sourceRange()

        torch._C._jit_pass_transform_conv1d_to_conv2d(model.graph)

        for node in model.graph.nodes():
            if node.kind() == "aten::conv2d":
                source_range_2 = node.sourceRange()

        FileCheck().check("aten::conv2d").run(model.graph)
        check_not = ["aten::conv1d"]
        for cn in check_not:
            FileCheck().check_not(cn).run(model.graph)

        # make sure it runs
        self.assertTrue(source_range_1 == source_range_2)
        model(x)
