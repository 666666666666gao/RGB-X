"""Task-adaptive AdamW candidates; optimizer statistics commit once per update."""
import torch


class TaskAdaptiveAdamW:
    def __init__(self, named_parameters, lr, weight_decay, clip_norm):
        named_parameters = list(named_parameters)
        self.names = [name for name, parameter in named_parameters if parameter.requires_grad]
        self.parameters = [parameter for name, parameter in named_parameters if parameter.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.shared = ['.experts.' not in name for name in self.names]
        self.state = []
        self.offsets = [0]
        for parameter, shared in zip(self.parameters, self.shared):
            tasks = 3 if shared else 1
            self.state.append(dict(m=torch.zeros((tasks,) + parameter.shape, device=parameter.device),
                                   v=torch.zeros((tasks,) + parameter.shape, device=parameter.device),
                                   steps=[0] * tasks))
            self.offsets.append(self.offsets[-1] + parameter.numel())

    def parameter_manifest(self):
        return [dict(name=name, shape=list(parameter.shape), numel=parameter.numel(),
                     category='router' if name.endswith(('w_gate', 'w_noise')) else ('shared_content' if shared else 'routed_expert'),
                     task_states=3 if shared else 1, risk_corrected=shared)
                for name, parameter, shared in zip(self.names, self.parameters, self.shared)]

    @torch.no_grad()
    def propose(self, gradients, active):
        joint = gradients.mean(0)
        coefficient = min(1.0, self.clip_norm / (joint.norm().item() + 1e-6))
        gradients = gradients * coefficient
        displacement = []
        for index, (parameter, shared, state) in enumerate(zip(self.parameters, self.shared, self.state)):
            start, end = self.offsets[index:index + 2]
            task_gradients = gradients[:, start:end].view((3,) + parameter.shape)
            if not active[:, index].any().item():
                displacement.append(torch.zeros_like(parameter).flatten())
                continue
            if shared:
                direction = torch.zeros_like(parameter)
                for task in range(3):
                    if not active[task, index].item():
                        continue
                    self._accumulate(state, task, task_gradients[task])
                    direction.add_(self._direction(state, task), alpha=1 / 3)
            else:
                self._accumulate(state, 0, task_gradients.mean(0))
                direction = self._direction(state, 0)
            displacement.append((-self.lr * direction - self.lr * self.weight_decay * parameter).flatten())
        delta = torch.cat(displacement)
        assert torch.isfinite(delta).all()
        return delta, dict(joint_grad_norm_before_clip=joint.norm().item(), gradient_clip_coefficient=coefficient)

    def _accumulate(self, state, task, gradient):
        state['steps'][task] += 1
        state['m'][task].mul_(self.beta1).add_(gradient, alpha=1 - self.beta1)
        state['v'][task].mul_(self.beta2).addcmul_(gradient, gradient, value=1 - self.beta2)

    def _direction(self, state, task):
        step = state['steps'][task]
        mean = state['m'][task] / (1 - self.beta1 ** step)
        variance = state['v'][task] / (1 - self.beta2 ** step)
        return mean / (variance.sqrt() + self.eps)

    @torch.no_grad()
    def values(self):
        return torch.cat([parameter.detach().flatten() for parameter in self.parameters])

    @torch.no_grad()
    def assign(self, vector):
        for index, parameter in enumerate(self.parameters):
            start, end = self.offsets[index:index + 2]
            parameter.copy_(vector[start:end].view_as(parameter))

    def shared_mask(self):
        return torch.cat([torch.full((parameter.numel(),), shared, dtype=torch.bool, device=parameter.device)
                          for parameter, shared in zip(self.parameters, self.shared)])

    def state_dict(self):
        return dict(names=self.names, lr=self.lr, weight_decay=self.weight_decay, clip_norm=self.clip_norm,
                    beta1=self.beta1, beta2=self.beta2, eps=self.eps, state=self.state)

    def load_state_dict(self, value):
        assert value['names'] == self.names
        self.lr = value['lr']
        self.state = value['state']
