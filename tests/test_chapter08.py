import unittest
from fractions import Fraction
import numpy as np
from nonlinear_workbook import chapter08 as c


class Chapter08Tests(unittest.TestCase):
    def test_cantor_exact(self):
        self.assertEqual(c.cantor_intervals(1),((Fraction(0),Fraction(1,3)),(Fraction(2,3),Fraction(1))))
        for depth in range(7):
            intervals=c.cantor_intervals(depth)
            self.assertEqual(len(intervals),2**depth)
            self.assertEqual(sum((b-a for a,b in intervals),Fraction(0)),Fraction(2,3)**depth)
            self.assertTrue(all(b-a==Fraction(1,3)**depth for a,b in intervals))
            self.assertTrue(all(intervals[i][1]<intervals[i+1][0] for i in range(len(intervals)-1)))
        self.assertEqual(c.cantor_endpoints(c.cantor_intervals(2)).shape,(8,1))

    def test_dragon_and_koch_geometry(self):
        for depth in range(9):
            dragon=c.dragon(depth)
            self.assertEqual(len(dragon),2**depth+1); self.assertEqual(dragon[0],0); self.assertEqual(dragon[-1],1)
            np.testing.assert_allclose(abs(np.diff(dragon)),2**(-depth/2),atol=1e-12,rtol=0)
        for depth in range(6):
            curve=c.koch_curve(depth); snow=c.koch_snowflake(depth)
            self.assertEqual(len(curve),4**depth+1); self.assertEqual(len(snow),3*4**depth+1)
            self.assertAlmostEqual(c.polyline_length(curve),(4/3)**depth,places=12)
            self.assertAlmostEqual(c.polyline_length(snow),3*(4/3)**depth,places=11)
            expected=np.sqrt(3)/4*(1+3/5*(1-(4/9)**depth))
            self.assertAlmostEqual(c.polygon_area(snow),expected,places=12)
            self.assertGreater(c.polygon_area(snow,signed=True),0)
        reverse=c.koch_snowflake(2,vertices=[0,complex(.5,np.sqrt(3)/2),1])
        self.assertLess(c.polygon_area(reverse,signed=True),0)
        self.assertAlmostEqual(c.polygon_area(reverse),c.polygon_area(c.koch_snowflake(2)))

    def test_gasket_and_ifs(self):
        area=np.sqrt(3)/4
        for depth in range(5):
            triangles=c.sierpinski_triangles(depth)
            self.assertEqual(triangles.shape,(3**depth,3,2))
            np.testing.assert_allclose(np.linalg.norm(triangles[:,1]-triangles[:,0],axis=1),2.**(-depth),atol=1e-14)
            self.assertAlmostEqual(sum(c.polygon_area(t[:,0]+1j*t[:,1]) for t in triangles),area*(3/4)**depth,places=12)
        a,b=c.gasket_ifs(); points=c.deterministic_ifs(a,b,[[0,0]],depth=1)
        np.testing.assert_array_equal(points,b)
        np.testing.assert_array_equal(c.contraction_norms(a,b),[.5]*3)
        p=c.gasket_ifs(printed=True)[1]; np.testing.assert_array_equal(p,[[1,1],[1,50],[50,50]])
        args=dict(probabilities=[1/3]*3,initial_point=[0,0],samples=100,burn_in=10,seed=8)
        first=c.chaos_game(a,b,**args); second=c.chaos_game(a,b,**args)
        np.testing.assert_array_equal(first.points,second.points); np.testing.assert_array_equal(first.map_indices,second.map_indices)

    def test_escape_boundaries_and_reference(self):
        result=c.mandelbrot(np.array([0,-1,-2,.5,3]),max_iterations=30)
        np.testing.assert_array_equal(result.escaped,[False,False,False,True,True])
        self.assertEqual(result.counts[-1],1)
        self.assertTrue(c.mandelbrot([3],max_iterations=1).escaped[0])
        self.assertFalse(c.mandelbrot([2],max_iterations=1).escaped[0])
        self.assertTrue(c.mandelbrot([2],max_iterations=2).escaped[0])
        julia=c.filled_julia([0,1,1j,2,3],c=0,max_iterations=10)
        np.testing.assert_array_equal(julia.escaped,[False,False,False,True,True]); self.assertEqual(julia.counts[-1],0)
        grid=np.linspace(-2,1,11)[:,None]+1j*np.linspace(-1,1,9)
        vector=c.mandelbrot(grid,max_iterations=40)
        for index in np.ndindex(grid.shape):
            z=0j; count=0
            while count<40 and abs(z)<=2: z=z*z+grid[index]; count+=1
            self.assertEqual(vector.counts[index],count); self.assertEqual(vector.escaped[index],abs(z)>2)
        conjugate=c.mandelbrot(grid.conjugate(),max_iterations=40)
        np.testing.assert_array_equal(vector.counts,conjugate.counts)
        parameter=-.12256117+.74486177j
        julia=c.filled_julia(grid,c=parameter,max_iterations=40)
        for index in np.ndindex(grid.shape):
            z=complex(grid[index]); count=0; radius=max(2,abs(parameter))
            while count<40 and abs(z)<=radius: z=z*z+parameter; count+=1
            self.assertEqual(julia.counts[index],count)
            self.assertEqual(julia.escaped[index],abs(z)>radius)
        nonfinite=c.filled_julia([1e200],c=1e300,max_iterations=2)
        self.assertTrue(nonfinite.nonfinite[0]); self.assertFalse(nonfinite.escaped[0])

    def test_weierstrass_and_tail(self):
        x=np.linspace(-1,1,100)
        result=c.weierstrass(x,s=1.5,lam=4,terms=3)
        np.testing.assert_allclose(result.values,.5*np.sin(4*x)+.25*np.sin(16*x)+.125*np.sin(64*x),atol=1e-15)
        deeper=c.weierstrass(x,s=1.5,lam=4,terms=12)
        self.assertLessEqual(np.max(abs(deeper.values-result.values)),result.tail_bound)
        self.assertEqual(result.tail_bound,.125)
        self.assertEqual(c.weierstrass([0],s=1.5,lam=4,terms=40).frequencies[-1],4.**40)

    def test_levy_distribution_sampling(self):
        result=c.levy_flight(steps=10000,lam=2,b=2,max_level=12,seed=20261014)
        repeat=c.levy_flight(steps=10000,lam=2,b=2,max_level=12,seed=20261014)
        np.testing.assert_array_equal(result.levels,repeat.levels)
        np.testing.assert_array_equal(np.diff(result.positions),result.increments)
        np.testing.assert_array_equal(result.increments,result.signs*2.**result.levels)
        self.assertAlmostEqual(result.probabilities.sum(),1,places=14)
        self.assertEqual(result.omitted_probability,2.**(-13))
        self.assertAlmostEqual(float(c.levy_characteristic(0,lam=2,b=2,max_level=12)),1,places=14)
        counts=np.bincount(result.levels,minlength=13)
        for count,p in zip(counts,result.probabilities):
            if 10000*p>=25: self.assertLess(abs(count-10000*p),5*np.sqrt(10000*p*(1-p)))

    def test_dimension_fit_reuse(self):
        for pieces,scale,target in ((2,3,np.log(2)/np.log(3)),(3,2,np.log(3)/np.log(2)),(4,3,np.log(4)/np.log(3))):
            depth=np.arange(1,7); fit=c.fit_dimension(scale**(-depth.astype(float)),pieces**depth,fit_slice=slice(0,6),kind='box')
            self.assertAlmostEqual(fit.slope,target,places=12)
            self.assertLess(max(abs(fit.residuals)),1e-12)
        points=np.column_stack((np.linspace(0,1,100),np.zeros(100)))
        boxes=c.box_counts(points,scales=[.1,.2],origin=[0,0]); self.assertEqual(len(boxes.counts),2)
        pairs=c.correlation_integral(points,radii=[.1,.2],theiler=0); self.assertEqual(pairs.eligible_pairs,4950)

    def test_invalid_and_allocation_guards(self):
        for call in (lambda:c.cantor_intervals(-1),lambda:c.dragon(50,max_elements=100),lambda:c.sierpinski_triangles(20,max_elements=100),lambda:c.koch_curve(10,max_elements=100),lambda:c.weierstrass([1],s=2,lam=4,terms=2),lambda:c.weierstrass([1],s=1.5,lam=1e300,terms=3),lambda:c.levy_distribution(lam=2,b=1e300,max_level=3),lambda:c.mandelbrot([np.nan],max_iterations=10),lambda:c.filled_julia([0],c=0,max_iterations=0),lambda:c.fit_dimension([1,.5],[1,2],fit_slice=slice(None),kind='box')):
            with self.assertRaises(ValueError): call()
        a,b=c.gasket_ifs()
        with self.assertRaises(ValueError): c.chaos_game(a,b,probabilities=[1,1,1],initial_point=[0,0],samples=10,burn_in=0,seed=0)
