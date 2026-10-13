import unittest
import numpy as np
from nonlinear_workbook.chapter01 import *

class StageBTests(unittest.TestCase):
    def test_analytic_jacobians_and_update_order(self):
        point=np.array([.3,.4]); h=1e-6
        pairs=[(henon_step,henon_jacobian),(lozi_step,lozi_jacobian),
               (standard_step,standard_jacobian),(ikeda_step,ikeda_jacobian),
               (coupled_logistic_step,coupled_logistic_jacobian),(rtn_step,rtn_jacobian)]
        for step,jac in pairs:
            finite_difference=np.column_stack([(step(point+h*np.eye(2)[i])-step(point-h*np.eye(2)[i]))/(2*h) for i in range(2)])
            np.testing.assert_allclose(jac(point),finite_difference,atol=1e-8,rtol=1e-8,err_msg=step.__name__)
        new=rtn_step(point,r=1.0834)
        self.assertAlmostEqual(new[1],1.0834*(3*new[0]+1)*point[1]*(1-point[1]))
        with self.assertRaises(ValueError): lozi_jacobian([0,.1])
        with self.assertRaises(ValueError): henon_step([1,2,3])
        self.assertAlmostEqual(np.linalg.det(standard_jacobian(point,k=1.2)),1,places=13)
        state=standard_step([100.,1.],k=0)
        self.assertEqual(state[0],100); self.assertTrue(0<=state[1]<2*np.pi)

    def test_fixed_points_and_classifications(self):
        a,b=1.4,.3
        roots=[((b-1)+sign*np.sqrt((1-b)**2+4*a))/(2*a) for sign in (-1,1)]
        for x in roots:
            point=[x,b*x]; np.testing.assert_allclose(henon_step(point),point,atol=1e-14)
            _,kind=fixed_point_classification(henon_jacobian,point)
            self.assertEqual(kind,'saddle')
        for matrix,expected in [(np.diag([.5,.8]),'sink'),(np.diag([2,3]),'source'),(np.diag([.5,2]),'saddle'),(np.diag([1,.5]),'nonhyperbolic')]:
            self.assertEqual(fixed_point_classification(lambda _:matrix,[0,0])[1],expected)

    def test_known_qr_linear_and_henon_determinant(self):
        matrix=np.array([[1.01,.03],[0,.99]])
        result=known_lyapunov_spectrum(np.zeros((100,2)),lambda _:matrix,qr_burn_in=5)
        np.testing.assert_allclose(result.exponents,np.log([1.01,.99]),atol=1e-13)
        self.assertEqual(result.accumulated_steps,95)
        scaled=known_lyapunov_spectrum(np.zeros((100,2)),lambda _:matrix,sample_interval=.5)
        np.testing.assert_allclose(scaled.exponents,2*np.log([1.01,.99]),atol=1e-13)
        orbit=iterate(henon_step,[.1,.3],n=10000,burn_in=1000)
        self.assertEqual(orbit.status,'completed')
        henon=known_lyapunov_spectrum(orbit.values,henon_jacobian,qr_burn_in=100)
        self.assertAlmostEqual(henon.exponents.sum(),np.log(.3),delta=1e-11)
        self.assertAlmostEqual(henon.exponents[0],.42,delta=.03)
        with self.assertRaises(ValueError): known_lyapunov_spectrum(np.zeros((10,2)),lambda _:np.diag([0,1]))
        with self.assertRaises(ValueError): known_lyapunov_spectrum(np.zeros((10,2)),lambda _:np.eye(2),qr_burn_in=10)

    def test_uncoupled_logistic_spectra(self):
        orbit=iterate(lambda x:coupled_logistic_step(x,r=4,coupling=0),[.618,.317],n=10000,burn_in=1000)
        result=known_lyapunov_spectrum(orbit.values,lambda x:coupled_logistic_jacobian(x,r=4,coupling=0))
        targets=[scalar_lyapunov(orbit.values[:,j],logistic_derivative) for j in (0,1)]
        np.testing.assert_allclose(result.exponents,targets,atol=1e-12)
        np.testing.assert_allclose(result.exponents,np.log(2),atol=.005)

    def test_pair_count_fixtures_and_exclusion(self):
        points=[0.,1.,1.,3.]; radii=[.1,1.,2.,3.]
        r=correlation_integral(points,radii=radii)
        np.testing.assert_array_equal(r.counts,[1,3,5,6]); self.assertEqual(r.eligible_pairs,6)
        r=correlation_integral(points,radii=radii,theiler=1)
        np.testing.assert_array_equal(r.counts,[0,1,2,3]); self.assertEqual(r.eligible_pairs,3)
        with self.assertRaises(ValueError): correlation_integral(points,radii=radii,theiler=10)
        with self.assertRaises(ValueError): correlation_integral(points,radii=[0])

    def test_line_square_dimensions_and_grid_origin(self):
        coordinate=(np.arange(64)+.5)/64
        scales=[.25,.125,.0625]; line=np.column_stack([coordinate,np.zeros(64)])
        xx,yy=np.meshgrid(coordinate,coordinate); square=np.column_stack([xx.ravel(),yy.ravel()])
        for points,target in [(line,1.),(square,2.)]:
            result=box_counts(points,scales=scales,origin=[0,0])
            fit=fit_scaling(result.scales,result.counts,fit_slice=slice(0,3),inverse_scale=True)
            self.assertAlmostEqual(fit.slope,target,places=12)
        shifted=box_counts(line,scales=scales,origin=[.001,0])
        self.assertTrue(np.all(shifted.counts>0))
        self.assertAlmostEqual(fit_scaling([1,2,4],[1,4,16],fit_slice=slice(0,3)).slope,2,places=12)
        with self.assertRaises(ValueError): fit_scaling([1,2],[0,1],fit_slice=slice(0,2))
        with self.assertRaises(ValueError): fit_scaling([1,2],[1,2],fit_slice=slice(0,3))

    def test_complex_and_vector_newton(self):
        for root in [1.,-1.,1j,-1j]:
            result=newton(lambda z:z**4-1,lambda z:4*z**3,root*1.1)
            self.assertEqual(result.status,'converged'); self.assertAlmostEqual(abs(result.root-root),0,places=10)
        basin=complex_newton_basins([-1,0,1],[-1,0,1])
        self.assertEqual(basin.labels[1,2],0); self.assertEqual(basin.labels[1,0],1)
        self.assertEqual(basin.labels[2,1],2); self.assertEqual(basin.labels[0,1],3)
        self.assertEqual(basin.statuses[1,1],'singular_derivative')
        f=lambda p:np.array([p@p-1,p[0]-p[1]])
        jac=lambda p:np.array([[2*p[0],2*p[1]],[1,-1]])
        for initial,sign in [([.7,.8],1),([-.7,-.8],-1)]:
            result=vector_newton(f,jac,initial)
            self.assertEqual(result.status,'converged')
            np.testing.assert_allclose(result.root,sign/np.sqrt(2),atol=1e-12)
        self.assertEqual(vector_newton(f,jac,[0,0]).status,'singular_jacobian')
        self.assertEqual(newton(lambda z:z**4-1,lambda z:4*z**3,.001,escape_radius=10).status,'escaped')

    def test_finite_horizon_labels(self):
        scan=boundedness_scan([2.5,4],[0,.5],horizon=30,escape_radius=20)
        self.assertEqual(scan.labels[0,0],'completed')
        self.assertEqual(scan.labels[1,1],'escaped')
        self.assertEqual(scan.horizon,30)
        self.assertLess(scan.updates[1,1],30)
        with self.assertRaises(ValueError): boundedness_scan([3],[0],x0=[1])

if __name__=='__main__': unittest.main()
