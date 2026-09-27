# Installation
Create the conda environment and install torch
```bash
conda create -n mtl python=3.9.7
conda activate mtl
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install the repo and:
```bash
cd repo_path
pip install -e .
```

# Download dataset

[CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)

[Office-31](https://github.com/median-research-group/LibMTL/tree/main/examples/office)

[NYU-v2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0)

[CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0) (the small version)
# CelebA results
see  celeba_mtl_result.md
# NSGA-II code
you should put the code in  deap.algorithms:

```python
def eaNSGA2(population, toolbox, cxpb, mutpb, ngen, npop, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fitness in zip(population, fitnesses):
        ind.fitness.values = fitness

    # rank 1 pareto set
    if halloffame is not None:
        halloffame.update(population)


    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, **record)
    if verbose:
        print(logbook.stream)


    fronts = tools.emo.sortNondominated(population, k=npop, first_front_only=False)
    for i, front in enumerate(fronts):     
        for ind in front:
            ind.fitness.values = i + 1,   

    offspring_sl = toolbox.selectGen1(population, npop)

    offspring = varAnd(offspring_sl, toolbox, cxpb, mutpb)

    for gen in range(1, ngen):

        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fitness in zip(offspring, fitnesses):
            ind.fitness.values = fitness


        combinedPop = population + offspring


        fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
        for ind, fitness in zip(combinedPop, fitnesses):
            ind.fitness.values = fitness

        print("-" * 50 + "{}".format(gen) + "-" * 50)

        fronts = tools.emo.sortNondominated(combinedPop, len(combinedPop))
    
        population = tools.selNSGA2(combinedPop, k=npop, nd='standard')

        matingpool = toolbox.select(population, npop)
        #print(rept_rate(population))

        if halloffame is not None:
            halloffame.update(population)


        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, **record)
        if verbose:
            print(logbook.stream)
 
        offspring = varAnd(matingpool, toolbox, cxpb, mutpb)

    return population,halloffame
```
# Compile function
Replace the compile function in deap.gp with:
```python
######################################
# GP Tree compilation functions      #
######################################
def compile(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                a string of Python code or any object that when
                converted into string produced a valid Python code
                expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
            or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        # args = ",".join(arg for arg in pset.arguments)
        for g in reversed(pset.arguments):
            if code.find(g) != -1:
                index = g[1:]
                replace_str = "list" + "[" + index + "]"
                # print(str)
                code = code.replace(g, replace_str)
        code = "lambda list: {code}".format(code=code)
    try:
        # print(code)
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort.").with_traceback(traceback)
```

⚠️ **Code Status** ⚠️

Our method has undergone multiple development iterations. We have released the installation instructions, dataset information, and the key modifications to the DEAP framework used in our experiments. To ensure that the code is well organized, the complete implementation of EvoGrad will be released within one week.
