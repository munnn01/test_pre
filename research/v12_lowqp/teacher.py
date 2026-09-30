"""Frozen ImageNet 2D teacher; never constructs an action analyzer."""
import hashlib
from pathlib import Path
import torch
from .policy import rgb112,dense_feature_distance

def state_hash(net):
    h = hashlib.sha256()
    for key,tensor in sorted(net.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        h.update(f'{key}:{array.dtype}:{array.shape}:'.encode())
        h.update(array.tobytes())
    return h.hexdigest()

class Teacher:
    def __init__(self,spec):
        from torchvision.models import resnet18,ResNet18_Weights
        if (spec['name']!='resnet18' or spec['weights']!='IMAGENET1K_V1'
            or spec['layers']!=['layer2','layer3'] or spec['input_size']!=112
            or spec['mean']!=[.485,.456,.406] or spec['std']!=[.229,.224,.225]):
            raise ValueError('teacher specification changed')
        weights = ResNet18_Weights.IMAGENET1K_V1
        if weights.url!=spec['url']:
            raise ValueError('teacher weight URL changed')
        state = weights.get_state_dict(progress=False,check_hash=True)
        path = Path(torch.hub.get_dir())/'checkpoints'/spec['url'].rsplit('/',1)[-1]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['checkpoint_sha256']:
            raise ValueError('teacher checkpoint hash mismatch')
        self.net = resnet18(weights=None)
        self.net.load_state_dict(state,strict=True)
        if state_hash(self.net)!=spec['state_sha256']:
            raise ValueError('teacher canonical state hash mismatch')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.eval().requires_grad_(False).to(self.device)
        self.spec = spec

    @torch.inference_mode()
    def features(self,clip):
        x = rgb112(clip).to(self.device)
        x = (x-x.new_tensor(self.spec['mean']).view(1,3,1,1))/x.new_tensor(self.spec['std']).view(1,3,1,1)
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer1(x)
        a = n.layer2(x)
        return (a,n.layer3(a))

    def score(self,source_features,decoded):
        return dense_feature_distance(source_features,self.features(decoded))

    def synchronize(self):
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
