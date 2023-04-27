module CartesianGeneticProgramming

import YAML
using Cambrian
import JSON

include("functions.jl")
include("config.jl")
include("individual.jl")
include("process.jl")
include("mutation.jl")
include("crossover.jl")
include("evolution.jl")
include("NSGA2_evolution.jl")
include("NSGA3_evolution.jl")
include("NS_evolution.jl")

end
