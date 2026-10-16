"""Chapter 13: explicit fuzzy membership, arithmetic and rule inference."""
from .sets import (FuzzySet,triangle,trapezoid,rectangle,ramp,gaussian,logistic,
                   spread,combine,complement,hedge,contrast,entropy,weighted_mean,
                   cartesian_min,gaussian_compatibility,compare)
from .arithmetic import extension,CutIntervals,Triangular,LRInterval
from .inference import Rule,infer,singleton_output
