import pytest
import torch
from dtu_mlops_garu.model import Model1, Model2

@pytest.mark.parametrize("model_class", [Model1, Model2])
class MyTestClass:
    def test_model_input(model_class):
        model = model_class(dropout = 0.5)

        with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
            model(torch.randn(1,2,3))
        with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
            model(torch.randn(1,1,28,29))

        y = model(torch.randn(1,1,28,28))
        assert y.shape == (1, 10)

        x = torch.rand(64, 1, 28, 28)
        y = model(x)
        assert y.shape == (64, 10)
    
