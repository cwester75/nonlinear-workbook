import unittest
from fractions import Fraction
import itertools
import numpy as np
from nonlinear_workbook import chapter13 as c


class SetsTests(unittest.TestCase):
    def test_membership_landmarks(self):
        np.testing.assert_array_equal(c.triangle([-1,0,1,2,3],0,1,2),[0,0,1,0,0])
        np.testing.assert_array_equal(c.triangle([-1,0,1],0,0,0),[0,1,0])
        np.testing.assert_array_equal(c.trapezoid([-1,0,1,2,3,4],0,1,2,3),[0,0,1,1,0,0])
        np.testing.assert_array_equal(c.trapezoid([-1,0,1,2],0,0,1,1),[0,1,1,0])
        np.testing.assert_array_equal(c.rectangle([-1,0,1,2],0,1),[0,1,1,0])
        np.testing.assert_allclose(c.ramp([-1,0,.5,1,2],0,1),[0,0,.5,1,1],atol=1e-12)
        self.assertEqual(float(c.gaussian(3,3,.2)),1)
        np.testing.assert_allclose(c.logistic([-1e308,0,1e308],0,1e-308),[0,.5,1],atol=1e-12)
    def test_finite_extreme_memberships(self):
        # The standardized distance is exactly +/-2 although subtraction overflows.
        import warnings
        with warnings.catch_warnings(), np.errstate(all='raise'):
            warnings.simplefilter('error',RuntimeWarning)
            for sign in (-1.,1.):
                x=sign*1e308;center=-sign*1e308
                self.assertAlmostEqual(float(c.gaussian(x,center,1e308)),np.exp(-2),places=12)
                self.assertAlmostEqual(float(c.logistic(x,center,1e308)),1/(1+np.exp(-2*sign)),places=12)
            # Equal huge values must not form inf-inf when divided by a tiny width.
            self.assertEqual(float(c.gaussian(1e308,1e308,1e-308)),1.)
            self.assertEqual(float(c.logistic(1e308,1e308,1e-308)),.5)
            # Preserve cancellation accuracy for adjacent finite same-sign values.
            center=1e308;width=np.spacing(center);x=np.nextafter(center,np.inf)
            self.assertAlmostEqual(float(c.gaussian(x,center,width)),np.exp(-.5),places=12)
            self.assertAlmostEqual(float(c.logistic(x,center,width)),1/(1+np.exp(-1)),places=12)
            values=np.array([-1e308,0,1e308]);centers=-values
            np.testing.assert_allclose(c.gaussian(values,centers,1e308),[np.exp(-2),1,np.exp(-2)],rtol=0,atol=1e-12)
            np.testing.assert_allclose(c.logistic(values,centers,1e308),[1/(1+np.exp(2)),.5,1/(1+np.exp(-2))],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(c.gaussian(values,0,1e-308),[0,1,0])
            np.testing.assert_array_equal(c.logistic(values,0,1e-308),[0,.5,1])
    def test_spreads(self):
        for kind in ('linear','exponential','gaussian','polynomial'):
            y=c.spread([0,.5,1,2],kind=kind)
            self.assertEqual(y[0],1);self.assertTrue(np.all(np.diff(y)<=0))
        interval=c.LRInterval(100,120,30,10)
        np.testing.assert_allclose(interval.membership([70,85,100,110,120,125,130]),[0,.5,1,1,1,.5,0],atol=1e-12)
        np.testing.assert_array_equal(c.LRInterval(1,1,0,0).membership([0,1,2]),[0,1,0])
    def test_set_semantics(self):
        s=c.FuzzySet([1,2,3],[0,.5,.5]);self.assertEqual(s.cardinality,1);self.assertEqual(s.relative_cardinality,1/3)
        np.testing.assert_array_equal(s.support,[2,3]);np.testing.assert_array_equal(s.alpha_cut(.5),[2,3]);np.testing.assert_array_equal(s.alpha_cut(0),[1,2,3])
        np.testing.assert_array_equal(s.top_beta(1),[2]);np.testing.assert_array_equal(s.top_beta(0),[])
        self.assertEqual(s.extremum(ties='first'),2);self.assertEqual(s.extremum(ties='last'),3);self.assertEqual(s.extremum(ties='mean'),2.5)
        np.testing.assert_array_equal(s.normalize().membership,[0,1,1]);self.assertTrue(s.subset(s.normalize()))
        self.assertEqual(s.at(1.5),.25);self.assertEqual(s.at(0,outside='zero'),0)
        with self.assertRaises(ValueError): s.at(0)
        with self.assertRaises(ValueError):s.operation(c.FuzzySet([0,1,2],[0,.5,.5]))
        with self.assertRaises(ValueError):s.membership[0]=1
    def test_crisp_exhaustive(self):
        for a,b in itertools.product((0.,1.),repeat=2):
            self.assertEqual(float(c.combine(a,b)),int(bool(a) and bool(b)))
            self.assertEqual(float(c.combine(a,b,operator='max')),int(bool(a) or bool(b)))
            self.assertEqual(float(c.complement(a)),1-a)
    def test_lattice_properties(self):
        rng=np.random.default_rng(20261016);a,b,d=rng.random((3,100))
        np.testing.assert_allclose(c.complement(c.combine(a,b)),c.combine(c.complement(a),c.complement(b),operator='max'),atol=1e-12)
        np.testing.assert_allclose(np.minimum(a,np.maximum(b,d)),np.maximum(np.minimum(a,b),np.minimum(a,d)),atol=1e-12)
        np.testing.assert_array_equal(c.complement(c.complement(a)),a)
        for op,base in [('fuzzy_and',np.minimum(a,b)),('fuzzy_or',np.maximum(a,b))]:
            np.testing.assert_allclose(c.combine(a,b,operator=op,gamma=1),base,atol=1e-12)
            np.testing.assert_allclose(c.combine(a,b,operator=op,gamma=0),(a+b)/2,atol=1e-12)
        for op in ('product','algebraic_sum','bounded_sum','bounded_difference'):
            y=c.combine(a,b,operator=op);self.assertTrue(np.all((y>=0)&(y<=1)))
        self.assertLess(float(c.combine(.5,.5,operator='max')),1) # No excluded-middle assertion.
    def test_entropy_hedge_mean(self):
        self.assertEqual(c.entropy([0,1]),0);self.assertAlmostEqual(c.entropy([.5]*4),4,places=12)
        a=np.array([.1,.4,.8]);self.assertAlmostEqual(c.entropy(a),c.entropy(1-a),places=12)
        np.testing.assert_allclose(c.hedge(a,2),a*a,atol=1e-12)
        np.testing.assert_allclose(c.contrast([0,.25,.5,.75,1]),[0,.125,.5,.875,1],atol=1e-12)
        np.testing.assert_allclose(c.weighted_mean([[0,1],[1,0]],[.2,.8]),[.8,.2],atol=1e-12)
        np.testing.assert_array_equal(c.cartesian_min([0,1],[.2,.8]),[[0,0],[.2,.8]])
    def test_centroids(self):
        self.assertAlmostEqual(c.FuzzySet(range(1,8),[.85,.7,.1,.44,.87,.2,.19]).centroid(),3.3402985074626863,places=12)
        errors=[]
        for n in (101,401,1601):
            x=np.linspace(0,3,n);s=c.FuzzySet(x,c.triangle(x,0,1,3));errors.append(abs(s.centroid(mode='continuous')-4/3))
        self.assertLess(errors[-1],2e-6);self.assertLess(errors[-1],errors[0]/100)
        self.assertAlmostEqual(c.FuzzySet([0,1,2],[1,1,1]).centroid(mode='continuous'),1,places=12)
    def test_gaussian_compatibility(self):
        r=c.gaussian_compatibility(1,2,4,1);self.assertAlmostEqual(r['value'],np.exp(-1),places=12)
        self.assertAlmostEqual(r['location'],3,places=12)
        self.assertAlmostEqual(c.gaussian_compatibility(1e308,1e308,-1e308,1e308)['value'],np.exp(-1),places=12)
        self.assertAlmostEqual(np.exp(-((r['location']-1)/2)**2),r['value'],places=12)
    def test_book_discrete_listing(self):
        a=c.FuzzySet([1,2,3,4,5],[.2,.4,1,.4,.8])
        self.assertAlmostEqual(a.centroid(mode='continuous'),3.260869565217391,places=12)
        self.assertAlmostEqual(a.cardinality,2.8,places=12)
        self.assertAlmostEqual(a.relative_cardinality,.56,places=12)
        self.assertAlmostEqual(c.entropy(a.membership),3.385757378684062,places=12)
    def test_book_pricing_language(self):
        x=np.arange(2000,5001,dtype=float)
        base=[c.ramp(x,2000,5000),c.ramp(x,2000,5000,increasing=False),c.gaussian(x,3250,300),c.logistic(x,2500,300),1-c.logistic(x,4500,300)]
        for powers,expected in [([2,1.5,.25,4,2],3649),([2,4,.25,4,4],3146)]:
            score=np.minimum.reduce([c.hedge(b,p) for b,p in zip(base,powers)])
            self.assertEqual(x[np.argmax(score)],expected)
        grid=np.linspace(0,10,200);slightly=c.FuzzySet(grid,c.ramp(grid,0,10)**.25)
        self.assertAlmostEqual(float(slightly.at(.1)),.3161205300436254,places=12)
    def test_book_shoes(self):
        looks=np.array([.7,1,.4,.7,.5]);comfort=np.array([.4,0,.8,.6,.9]);valid=np.array([1,0,1,1,0])
        score=np.minimum(valid,(looks+comfort)/2);self.assertEqual(int(np.argmax(score)),3)
        score=np.minimum(valid,(np.sqrt(looks)+comfort**2)/2);self.assertEqual(int(np.argmax(score)),2)
    def test_clipping_comparison_quasiconcavity(self):
        a=c.FuzzySet([0,1,2],[0,1,0]);self.assertTrue(a.sampled_quasiconcave())
        self.assertFalse(c.FuzzySet([0,1,2],[1,0,1]).sampled_quasiconcave())
        np.testing.assert_array_equal(a.clipped(.4).membership,[0,.4,0])
        np.testing.assert_array_equal(c.compare([0,.5,1],[.5,.5,.5],operator='gt'),[0,0,1])
        np.testing.assert_array_equal(c.spread([0,.5,1],kind='polynomial_outer',power=2),[1,.25,0])
    def test_invalid_sets(self):
        for x,m in [([],[]),([1,1],[0,1]),([1,0],[0,1]),([0,1],[-.1,1]),([0,1],[0,np.nan])]:
            with self.assertRaises(ValueError):c.FuzzySet(x,m)
        for fn in [lambda:c.triangle([0],2,1,3),lambda:c.FuzzySet(np.array([1+1j]),[1]),lambda:c.gaussian([0],0,0),lambda:c.logistic([0],0,-1),lambda:c.hedge([.5],0),lambda:c.entropy([.5],base=1),lambda:c.weighted_mean([[0],[1]],[0,0]),lambda:c.cartesian_min([0]*100,[1]*100,max_elements=10),lambda:c.FuzzySet([0],[0]).normalize(),lambda:c.FuzzySet([0],[0]).centroid()]:
            with self.assertRaises(ValueError):fn()


class ArithmeticTests(unittest.TestCase):
    def test_extension_book(self):
        self.assertEqual(c.extension([[-1,0,1,2]],[[.5,.8,1,.4]],lambda x:x*x),{0:.8,1:1.,4:.4})
        add=c.extension([[1,2,3,4],[3,4,5]],[[.2,1,.5,.3],[.9,1,.4]],lambda x,y:x+y)
        self.assertEqual(add,{4:.2,5:.9,6:1.,7:.5,8:.4,9:.3})
        mult=c.extension([[1,2,3],[2,3,4]],[[.3,1,.4],[.7,1,.2]],lambda x,y:x*y)
        self.assertEqual(mult,{2:.3,3:.3,4:.7,6:1.,8:.2,9:.4,12:.2})
    def test_extension_numpy_integer_overflow(self):
        fixtures=[(np.int8,100),(np.int16,30000),(np.int32,100000),
                  (np.int64,2**32),(np.uint64,2**64-1)]
        for dtype,value in fixtures:
            received=[]
            def square(x):
                received.append(type(x));return x*x
            with self.subTest(dtype=dtype),np.errstate(over='raise',invalid='raise'):
                result=c.extension([np.array([value],dtype=dtype)],[[1]],square)
                self.assertEqual(result,{value*value:1.})
                self.assertEqual(received,[int])
                self.assertTrue(all(type(k) is int for k in result))
        # Signed endpoints and mixed Fraction arithmetic also retain exact keys.
        values=np.array([-(2**63),2**63-1],dtype=np.int64)
        actual=c.extension([values,[Fraction(1,3)]],[[.4,1],[1]],lambda x,y:-x+y)
        self.assertEqual(actual,{Fraction(-(2**63-1))+Fraction(1,3):1.,Fraction(2**63)+Fraction(1,3):.4})
    def test_fraction_independent_oracle(self):
        x=[Fraction(1,3),Fraction(2,3)];y=[Fraction(1,2),Fraction(1,3)];m=[.4,1];n=[1,.2]
        actual=c.extension([x,y],[m,n],lambda a,b:a+b)
        expected={}
        for i,a in enumerate(x):
            for j,b in enumerate(y):expected[a+b]=max(expected.get(a+b,0),min(m[i],n[j]))
        self.assertEqual(actual,dict(sorted(expected.items())));self.assertTrue(all(isinstance(k,Fraction) for k in actual))
    def test_cut_arithmetic(self):
        levels=np.array([0,.25,.5,1]);a=c.Triangular(2,.5,.5);b=c.Triangular(3,.5,.5)
        ac=a.cuts(levels);bc=b.cuts(levels)
        np.testing.assert_allclose(ac.operation(bc,operator='add').bounds,a.add(b).cuts(levels).bounds,atol=1e-12)
        np.testing.assert_allclose(ac.operation(bc,operator='subtract').bounds,a.subtract(b).cuts(levels).bounds,atol=1e-12)
        for op,fn in [('multiply',lambda a,b:a*b),('divide',lambda a,b:a/b)]:
            r=ac.operation(bc,operator=op)
            for row,u,v in zip(r.bounds,ac.bounds,bc.bounds):np.testing.assert_allclose(row,[min(fn(i,j) for i in u for j in v),max(fn(i,j) for i in u for j in v)],atol=1e-12)
        signed=c.Triangular(-2,1,.5).cuts(levels).operation(ac,operator='multiply');self.assertTrue(np.all(signed.bounds<0))
    def test_triangular_signed_and_approximate(self):
        a=c.Triangular(2,.5,1);b=a.scale(-2)
        self.assertEqual(b,c.Triangular(-4,2,1));self.assertEqual(a.negate(),c.Triangular(-2,1,.5))
        signed=c.Triangular(-2,.5,1).approximate_product(c.Triangular(3,.2,.4))
        np.testing.assert_allclose([signed.mode,signed.left,signed.right],[-6,2.3,3.4],atol=1e-12)
        self.assertEqual(c.Triangular(2,.5,.5).approximate_product(c.Triangular(3,.5,.5)),c.Triangular(6,2.5,2.5))
        q=c.Triangular(2,.5,.5).approximate_divide(c.Triangular(3,.5,.5));self.assertAlmostEqual(q.left,11/42,places=12);self.assertAlmostEqual(q.right,.3,places=12)
        reciprocal=c.Triangular(-3,.5,.25).approximate_reciprocal()
        np.testing.assert_allclose([reciprocal.mode-reciprocal.left,reciprocal.mode+reciprocal.right],[-1/2.75,-1/3.5],atol=1e-12)
        np.testing.assert_array_equal(c.Triangular(2,0,0).membership([1,2,3]),[0,1,0])
    def test_product_compression_refinement(self):
        errors=[]
        for width in (.5,.25,.125):
            a=c.Triangular(2,width,width);b=c.Triangular(3,width,width);levels=np.linspace(0,1,41)
            exact=a.cuts(levels).operation(b.cuts(levels),operator='multiply').bounds
            approximate=a.approximate_product(b).cuts(levels).bounds
            errors.append(float(np.max(abs(exact-approximate))))
        np.testing.assert_allclose(errors,[.25,.0625,.015625],atol=1e-12)
    def test_trapezoidal_cuts(self):
        result=c.LRInterval(1,2,1,2).cuts([0,.5,1])
        np.testing.assert_allclose(result.bounds,[[0,4],[.5,3],[1,2]],atol=1e-12)
        with self.assertRaises(ValueError):c.LRInterval(1,2,1,2,left_kind='gaussian').cuts([0,1])
    def test_overflow_and_allocation(self):
        with self.assertRaises(ValueError):c.extension([[1]],[[1]],lambda x:float('inf'))
        with self.assertRaises(ValueError):c.cartesian_min([0],[1],max_elements=False)
        with self.assertRaises(ValueError):c.FuzzySet(range(1_000_001),range(1_000_001))
        with self.assertRaises(ValueError):c.CutIntervals(range(1_000_001),[])
        with self.assertRaises(ValueError):c.FuzzySet(range(10**12),range(10**12))
        with self.assertRaises(ValueError):c.CutIntervals(range(10**12),[])
        with self.assertRaises(ValueError):c.cartesian_min(range(10**12))
        with self.assertRaises(ValueError):c.extension([range(10**12)],[range(10**12)],lambda x:x)
        with self.assertRaises(ValueError):c.Triangular(1e308,0,0).scale(2)
    def test_arithmetic_invalid(self):
        for fn in [lambda:c.extension([[1,2]],[[0,1]],lambda x:x,max_combinations=1),lambda:c.extension([[1,1]],[[0,1]],lambda x:x),lambda:c.extension([[1]],[[1]],lambda x:np.inf),lambda:c.Triangular(0,1,1).approximate_reciprocal(),lambda:c.Triangular(1,-1,1),lambda:c.CutIntervals([0,1],[[0,1],[-1,2]]),lambda:c.Triangular(1,1,1).cuts([0,1]).operation(c.Triangular(0,1,1).cuts([0,1]),operator='divide')]:
            with self.assertRaises(ValueError):fn()


class InferenceTests(unittest.TestCase):
    def fixture(self):
        x=np.linspace(-2,2,101)
        return {'x':{'positive':lambda v:float(c.ramp(v,0,1))}}, {'negative':c.FuzzySet(x,c.triangle(x,-2,-1,0))},[c.Rule((('x','positive'),),'negative')]
    def test_rules_scalar_oracle(self):
        terms,outputs,rules=self.fixture()
        for method in ('clip','product'):
            r=c.infer({'x':.3},terms,outputs,rules,implication=method)
            expected=[min(.3,float(v)) if method=='clip' else .3*float(v) for v in outputs['negative'].membership]
            np.testing.assert_allclose(r['aggregate'],expected,atol=1e-12);self.assertAlmostEqual(r['value'],-1,places=12)
            self.assertEqual(r['status'],'completed');self.assertEqual(r['rules'][0]['strength'],.3)
    def test_connectives_aggregation(self):
        terms,outputs,_=self.fixture();terms['y']={'small':lambda x:.4}
        for op,target in [('min',.3),('max',.4),('product',.12)]:
            r=c.infer({'x':.3,'y':1},terms,outputs,[c.Rule((('x','positive'),('y','small')),'negative',op)])
            self.assertAlmostEqual(r['rules'][0]['strength'],target,places=12)
        r=c.infer({'x':.3},terms,outputs,[c.Rule((('x','positive'),),'negative'),c.Rule((('x','positive'),),'negative',weight=.5)])
        np.testing.assert_allclose(r['aggregate'],r['rules'][0]['membership'],atol=1e-12)
    def test_history_limit_and_rule_ties(self):
        terms,outputs,rules=self.fixture()
        with self.assertRaises(ValueError):c.infer({'x':1},terms,outputs,rules*10000)
        output={'negative':c.FuzzySet([-1,0,1],[1,0,1])}
        r=c.infer({'x':1},terms,output,rules,defuzzification='maximum',ties='last')
        self.assertEqual(r['value'],1)
    def test_unavailable(self):
        terms,outputs,rules=self.fixture()
        self.assertEqual(c.infer({'x':-1},terms,outputs,rules)['status'],'no_rule_fired')
        outputs={'negative':c.FuzzySet([-1,0,1],[0,0,0])}
        self.assertEqual(c.infer({'x':1},terms,outputs,rules)['status'],'zero_output_mass')
        self.assertIsNone(c.singleton_output([1,2],[0,0])['value']);self.assertEqual(c.singleton_output([1,2],[.5,.5])['value'],1.5)
    def test_invalid(self):
        terms,outputs,rules=self.fixture()
        for fn in [lambda:c.Rule((),'a'),lambda:c.Rule((1,),'a'),lambda:c.infer({'x':1},terms,outputs,['bad']),lambda:c.infer({'x':np.nan},terms,outputs,rules),lambda:c.infer({'y':1},terms,outputs,rules),lambda:c.infer({'x':1},terms,outputs,rules,implication='unknown')]:
            with self.assertRaises(ValueError):fn()

if __name__=='__main__':unittest.main()
