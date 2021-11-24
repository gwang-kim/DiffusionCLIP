import numpy as np
import torch


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def denoising_step(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   sampling_type='ddpm',
                   eta=0.0,
                   learn_sigma=False,
                   hybrid=False,
                   hybrid_config=None,
                   ratio=1.0,
                   out_x0_t=False,
                   ):

    # Compute noise and variance
    if type(models) != list:
        model = models
        et = model(xt, t)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = extract(logvars, t, xt.shape)
    else:
        if not hybrid:
            et = 0
            logvar = 0
            if ratio != 0.0:
                et_i = ratio * models[1](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += ratio * extract(logvars, t, xt.shape)
                et += et_i

            if ratio != 1.0:
                et_i = (1 - ratio) * models[0](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += (1 - ratio) * extract(logvars, t, xt.shape)
                et += et_i

        else:
            for thr in list(hybrid_config.keys()):
                if t.item() >= thr:
                    et = 0
                    logvar = 0
                    for i, ratio in enumerate(hybrid_config[thr]):
                        ratio /= sum(hybrid_config[thr])
                        et_i = models[i+1](xt, t)
                        if learn_sigma:
                            et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                            logvar_i = logvar_learned
                        else:
                            logvar_i = extract(logvars, t, xt.shape)
                        et += ratio * et_i
                        logvar += ratio * logvar_i
                    break

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)

    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    if out_x0_t == True:
        return xt_next, x0_t
    else:
        return xt_next


