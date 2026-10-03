import copy
import itertools
import operator
import random

import numpy as np
import torch
from deap import algorithms, base, creator, gp, tools

ngen = 50
CXPB = 0.8
MUTPB = 0.3

INTERMEDIATE_CLIP = 20.0
DIVISION_EPSILON = 1e-4


def _safe_tensor(value, clip=INTERMEDIATE_CLIP):
    return torch.nan_to_num(
        value,
        nan=0.0,
        posinf=clip,
        neginf=-clip,
    ).clamp(-clip, clip)


# def _execute_program(func, terminals):
#     """Support both standard DEAP and the server's list-input DEAP fork."""
#     code = getattr(func, "__code__", None)
#     if code is not None and code.co_argcount == 1:
#         dummy = torch.zeros_like(terminals[0])
#         return func([dummy, *terminals])
#     return func(*terminals)


@torch.no_grad()
def evograd_objevtives(ind, toolbox, G_normalized, G):
    func = toolbox.compile(expr=ind)
    d = _safe_tensor(func(*G_normalized.unbind(dim=1)))
    rates = G.T @ d

    total_rate = rates.sum()
    positive_rate_num = (rates > 0).sum()
    min_rate = rates.min()

    if not torch.isfinite(rates).all() or not torch.isfinite(total_rate):
        return -float("inf"), -float("inf"), -float("inf")

    return total_rate.item(), positive_rate_num.item(), min_rate.item()


def initial(feat_num):

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, feat_num), float, "g")

    def add(left, right):
        return _safe_tensor(left + right)


    def sub(left, right):
        return _safe_tensor(left - right)


    def mul(left, right):
        return _safe_tensor(left * right)


    def Div(left, right):
        sign = torch.where(right < 0,-torch.ones_like(right),torch.ones_like(right),)

        denominator = torch.where(right.abs() >= DIVISION_EPSILON,right,sign * DIVISION_EPSILON,)
        return _safe_tensor(left / denominator)


    pset.addPrimitive(add,[float, float],float,name="add",)
    pset.addPrimitive(sub,[float, float],float,name="sub",)
    pset.addPrimitive(mul,[float, float],float,name="mul",)
    pset.addPrimitive(Div,[float, float],float,name="Div",)
    creator.create("Fitnessmax", base.Fitness, weights=(1.0, 1.0, 1.0))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitnessmax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # EW ind
    terms = list(pset.arguments)
    
    while len(terms) > 1:
        terms = [
            f"add({terms[i]}, {terms[i + 1]})"
            if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    equal_weight = creator.Individual(gp.PrimitiveTree.from_string(terms[0], pset))

    toolbox.register("equal_weight", toolbox.clone, equal_weight)

    toolbox.register("selectGen1", tools.selTournament, tournsize=6)
    toolbox.register("select", tools.selTournamentDCD)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    pop = toolbox.population(n=500)
    pop[0] = toolbox.equal_weight()

    hof = tools.ParetoFront()
    # pareto_set = tools.ParetoFront()
    return pop, toolbox,hof

def _full_archive_front(
    archive,
    toolbox,
    full_inputs,
    full_G,
):
    evaluated = []

    for archived_ind in archive:
        individual = toolbox.clone(archived_ind)

        fitness = evograd_objevtives(individual,toolbox,full_inputs.detach(),full_G.detach(),)

        fitness_array = np.asarray(fitness,dtype=np.float64,)

        if not np.isfinite(fitness_array).all():
            continue

        individual.fitness.values = fitness
        evaluated.append(individual)

    if not evaluated:
        return []

    fronts = tools.emo.sortNondominated(
        evaluated,
        k=len(evaluated),
        first_front_only=True,
    )

    return distinct(fronts[0])

@torch.no_grad()
def evo(
    G_normalized, G, k, pop, toolbox, hof, nondominated_set_k,
    *, full_inputs,full_G, model, optimizer, shared_parameters,
    metric_batch, dataset, max_norm=1.0, metric_fn=None,
):


    toolbox.register("evaluate", evograd_objevtives,toolbox=toolbox,G_normalized=G_normalized.detach(),G=G.detach(),)

    evolve_now = k == 1 or k % 5 == 0

    if evolve_now:
        hof.clear()
        
        current_ngen = ngen if k == 1 else 2
        pop, _ = algorithms.eaNSGA2(pop, toolbox,CXPB, MUTPB, current_ngen, len(pop),stats=None,halloffame=None,verbose=True,)
        equal_weight = toolbox.equal_weight()

        if not any(str(ind) == str(equal_weight) for ind in pop):

            equal_weight.fitness.values = toolbox.evaluate(equal_weight)

            pop[-1] = equal_weight

        # fronts = tools.emo.sortNondominated(pop, len(pop))
        # candidates = distinct(fronts[0])

        # nondominated_set_k = candidates
        final_front = tools.emo.sortNondominated(pop,k=len(pop),first_front_only=True,)[0]

        nondominated_set_k = [toolbox.clone(ind) for ind in distinct(final_front)]

    # candidates = [toolbox.clone(ind) for ind in nondominated_set_k]
    # full_front = _full_archive_front(nondominated_set_k,toolbox,full_inputs,full_G,)

    sampled_front = _full_archive_front(
        nondominated_set_k,
        toolbox,
        G_normalized,
        G,
    )

    candidates = Diversity_Selection(sampled_front)

    # for ind in candidates:
    #     ind.fitness.values = toolbox.evaluate(ind)

    # current_pareto = tools.emo.sortNondominated(full_front, len(full_front))
    # candidates = distinct(current_pareto[0])

    # candidates = Diversity_Selection(full_front)
    direction, info = metric_guided_selection(
        candidates, toolbox, full_inputs, model, optimizer,
        shared_parameters, metric_batch, dataset,
        max_norm=max_norm, metric_fn=metric_fn,
    )
    return pop, direction, nondominated_set_k, info


def distinct(pareto_first_front):
    pareto_first_front_str = map(str, pareto_first_front)
    pareto_first_front_str = set(pareto_first_front_str)
    # for ind in pareto_first_front_str:
    #     print(ind)
    pareto_no_repeat = []
    for ind in pareto_first_front:
        if str(ind) in pareto_first_front_str:
            pareto_no_repeat.append(ind)
            pareto_first_front_str.remove(str(ind))
    return pareto_no_repeat


def Diversity_Selection(current_pareto):
    if len(current_pareto) <= 10:
        return current_pareto
    # max_firt_fitness = max(ind.fitness.values[0] for ind in current_pareto)
    # max_second_fitness = max(ind.fitness.values[1] for ind in current_pareto)
    # max_third_fitness = max(ind.fitness.values[2] for ind in current_pareto)
    # ck.append((max_firt_fitness, max_second_fitness, max_third_fitness))
    selected_inds = distinct([max(current_pareto, key=lambda ind: ind.fitness.values[i]) for i in range(3)])

    tools.emo.assignCrowdingDist(current_pareto)

    selected_programs = {str(ind) for ind in selected_inds}
    remain_inds = [
        ind for ind in current_pareto
        if str(ind) not in selected_programs
    ]
    remain_inds.sort(
        key=lambda ind: ind.fitness.crowding_dist,
        reverse=True,
    )

    remaining_slots =10 - len(selected_inds)
    selected_inds.extend(remain_inds[:remaining_slots])
    
    return selected_inds


@torch.no_grad()
def _set_shared_grad(shared_parameters, direction):
    offset = 0
    for param in shared_parameters:
        size = param.numel()
        param.grad = direction[offset:offset + size].reshape_as(param).clone()
        offset += size

def _optimizer_state_to_cpu(optimizer):
    source = optimizer.state_dict()
    copied_state = {}

    for parameter_id, state in source["state"].items():
        copied_state[parameter_id] = {
            key: (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            )
            for key, value in state.items()
        }

    return {
        "state": copied_state,
        "param_groups": copy.deepcopy(
            source["param_groups"]
        ),
    }


@torch.no_grad()
def metric_guided_selection(candidates,toolbox,full_inputs,model,optimizer,shared_parameters,metric_batch,dataset,max_norm=1.0,metric_fn=None,):
    shared_parameters = list(shared_parameters)
    parameters = list(model.parameters())

    device = parameters[0].device

    saved_grads = [None if parameter.grad is None else parameter.grad.detach().cpu().clone() for parameter in parameters]

    cpu_rng = torch.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()

    optimizer_state_cpu = _optimizer_state_to_cpu(optimizer)

    virtual_model = copy.deepcopy(model).to(device)

    original_parameters = list(model.parameters())
    virtual_parameters = list(virtual_model.parameters())

    parameter_map = {id(original): virtual for original, virtual in zip(original_parameters,virtual_parameters,)}

    virtual_param_groups = []

    for group in optimizer.param_groups:
        virtual_group = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}

        virtual_group["params"] = [parameter_map[id(parameter)] for parameter in group["params"]]

        virtual_param_groups.append(virtual_group)

    virtual_optimizer = optimizer.__class__(virtual_param_groups)

    virtual_optimizer.load_state_dict(copy.deepcopy(optimizer_state_cpu))

    virtual_shared_parameters = [parameter_map[id(parameter)] for parameter in shared_parameters]

    best_direction = None
    best_program = None
    best_score = -float("inf")
    scores = {}

    try:
        for ind in candidates:

            virtual_model.load_state_dict(model.state_dict())
            virtual_optimizer.load_state_dict(copy.deepcopy(optimizer_state_cpu))
            virtual_optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
            np.random.set_state(numpy_rng)
            random.setstate(python_rng)
            func = toolbox.compile(expr=ind)
            d = _safe_tensor(func(*full_inputs.unbind(dim=1)))

            if (d.shape != full_inputs.shape[:1] or not torch.isfinite(d).all()):
                continue

            if max_norm > 0:
                d_norm = torch.linalg.vector_norm(d)

                if not torch.isfinite(d_norm):
                    continue
            _set_shared_grad(virtual_shared_parameters,d,)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(virtual_shared_parameters,max_norm,error_if_nonfinite=True,)
            virtual_optimizer.step()
            virtual_model.eval()
            metrics = metric_fn(virtual_model,metric_batch,)

            if torch.is_tensor(metrics):
                metrics = (metrics.detach().cpu().numpy())

            metrics = np.asarray(metrics,dtype=np.float64,).reshape(-1)
            if not np.isfinite(metrics).all():
                continue

            score = _metric_score(metrics,dataset,full_inputs.shape[1],)

            scores[str(ind)] = score
            if (np.isfinite(score) and score > best_score):
                best_direction = d.detach().clone()
                best_program = str(ind)
                best_score = score

    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        for parameter, saved_grad in zip(parameters,saved_grads,):
            if saved_grad is None:
                parameter.grad = None
            else:
                parameter.grad = saved_grad.to(parameter.device)

        del virtual_optimizer
        del virtual_model
        del optimizer_state_cpu

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_direction, {
        "selected_program": best_program,
        "metric_score": best_score,
        "candidate_scores": scores,
    }

def _metric_score(metrics, dataset, n_tasks):
    metrics = np.asarray(metrics, dtype=np.float64).reshape(-1)

    if dataset in ("celeba", "office"):
        return float(metrics.mean())

    if dataset == "nyuv2":
        from experiments.nyuv2.utils import delta_p_fn
        baseline = stl_train_results["nyuv2"]
    elif dataset == "cityscapes":
        from experiments.cityscapes.utils import delta_p_fn
        baseline = stl_train_results["cityscapse"]
    else:
        raise ValueError("wrong dataset")

    return delta_p_fn(metrics, baseline=baseline)

stl_train_results = {
    'cityscapse': [0.8062, 0.9548, 0.0067, 21.57],
    'nyuv2': [0.8595, 0.9372, 0.2808, 0.1358, 18.3, 12.9, 0.4414, 0.738, 0.8318]
}
