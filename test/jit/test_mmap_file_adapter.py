import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.testing._internal.jit_utils import JitTestCase

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.weight = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)

class TestMmapFileAdapter(JitTestCase):
    def test_1(self):
        x = torch.FloatTensor([1, 3, 5])
        save_location = 'model.pkl'
        # save_location = 'pytorchmodel_saved.pt'

        m = MyModule()

        scr = torch.jit.script(m)
        scr = optimize_for_mobile(scr)
        scr._save_for_lite_interpreter(save_location)

        n = torch.jit.load(save_location)

        self.assertEqual(m(x), n(x))
        print('n(x): ' + str(n(x)))
        self.assertTrue(True)
