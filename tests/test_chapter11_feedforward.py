import unittest
from dataclasses import fields
import numpy as np
from nonlinear_workbook.chapter11 import *
from nonlinear_workbook.chapter11.fixtures import (BOOLEAN_INPUTS,XOR_TARGETS,XOR_HIDDEN_WEIGHTS,
    XOR_HIDDEN_BIAS,XOR_OUTPUT_WEIGHTS,XOR_OUTPUT_BIAS)


def book_initial():
    return MLPParameters(XOR_HIDDEN_WEIGHTS,XOR_HIDDEN_BIAS,XOR_OUTPUT_WEIGHTS,XOR_OUTPUT_BIAS)


class FeedforwardTests(unittest.TestCase):
    def test_perceptron_and_separator_and_xor_limit(self):
        np.testing.assert_array_equal(perceptron_predict(BOOLEAN_INPUTS,[1,1],bias=-1.5),[0,0,0,1])
        result=perceptron_train(BOOLEAN_INPUTS,[0,0,0,1],seed=20261012)
        self.assertEqual(result.status,'converged');np.testing.assert_array_equal(result.predictions,[0,0,0,1])
        self.assertLess(result.bias,0)
        points=np.array([[2,0],[1.5,1.5],[0,1],[1,0],[0,0]])
        separator=perceptron_train(points,[1,1,0,0,0])
        self.assertEqual(separator.status,'converged')
        failure=perceptron_train(BOOLEAN_INPUTS,XOR_TARGETS[:,0],max_epochs=100)
        self.assertEqual(failure.status,'max_epochs');self.assertGreater(failure.error_counts[-1],0)
        np.testing.assert_array_equal(xor_network(BOOLEAN_INPUTS),XOR_TARGETS[:,0])

    def test_all_sixteen_boolean_truth_tables(self):
        converged=[]
        for code in range(16):
            targets=np.array([(code >> (3-j)) & 1 for j in range(4)])
            result=perceptron_train(BOOLEAN_INPUTS,targets,max_epochs=300,seed=20261012)
            if code in (6,9):
                self.assertEqual(result.status,'max_epochs')
            else:
                self.assertEqual(result.status,'converged',f'Boolean code {code}')
                np.testing.assert_array_equal(result.predictions,targets)
                converged.append(code)
        self.assertEqual(len(converged),14)

    def test_every_mlp_weight_and_bias_gradient(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(3,2));targets=rng.uniform(size=(3,2))
        weights=mlp_initialize(2,hidden_width=3,n_outputs=2,seed=9)
        loss,gradient=mlp_loss_and_gradients(x,targets,weights,hidden_slope=1.3,output_slope=2.1)
        self.assertGreater(loss,0)
        h=1e-6
        for field in fields(weights):
            analytic=getattr(gradient,field.name);numeric=np.empty_like(analytic)
            for index in np.ndindex(analytic.shape):
                copied={f.name:getattr(weights,f.name).copy() for f in fields(weights)}
                copied[field.name][index]+=h
                plus=mlp_loss_and_gradients(x,targets,MLPParameters(**copied),hidden_slope=1.3,output_slope=2.1)[0]
                copied[field.name][index]-=2*h
                minus=mlp_loss_and_gradients(x,targets,MLPParameters(**copied),hidden_slope=1.3,output_slope=2.1)[0]
                numeric[index]=(plus-minus)/(2*h)
            np.testing.assert_allclose(analytic,numeric,atol=1e-6,rtol=1e-5,err_msg=field.name)

    def test_batch_update_uses_same_current_weights_and_mean_loss(self):
        initial=mlp_initialize(2,hidden_width=2,seed=5)
        x=BOOLEAN_INPUTS;targets=XOR_TARGETS
        before,gradient=mlp_loss_and_gradients(x,targets,initial)
        result=mlp_train(x,targets,initial=initial,learning_rate=.1,mode='batch',max_epochs=1,tolerance=1e-12)
        for field in fields(initial):
            np.testing.assert_allclose(getattr(result.weights,field.name),getattr(initial,field.name)-.1*getattr(gradient,field.name),atol=1e-14)
        self.assertLess(result.losses[-1],before)
        expected=.5*np.mean(np.sum((result.predictions-targets)**2,axis=1))
        self.assertAlmostEqual(result.losses[-1],expected,places=15)
        self.assertEqual(result.status,'max_epochs')

    def test_xor_training_book_fixture_and_repeatability(self):
        controls=dict(initial=book_initial(),hidden_slope=10,output_slope=10,learning_rate=.05,seed=20261012)
        result=mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,**controls)
        self.assertEqual(result.status,'converged')
        self.assertLess(np.max(np.abs(result.predictions-XOR_TARGETS)),.05)
        self.assertLessEqual(result.losses[-1],.00025)
        repeat=mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,**controls)
        np.testing.assert_array_equal(result.losses,repeat.losses)
        np.testing.assert_array_equal(result.predictions,repeat.predictions)
        np.testing.assert_array_equal(book_initial().hidden_weights,XOR_HIDDEN_WEIGHTS)

    def test_saturation_zero_epoch_and_invalid_inputs(self):
        saturated=MLPParameters(np.full((2,2),1000.),np.zeros(2),np.full((1,2),1000.),np.zeros(1))
        outputs=mlp_forward(BOOLEAN_INPUTS,saturated)
        self.assertTrue(np.all(np.isfinite(outputs)))
        loss,gradient=mlp_loss_and_gradients(BOOLEAN_INPUTS,XOR_TARGETS,saturated)
        self.assertTrue(np.isfinite(loss));self.assertTrue(np.all(np.isfinite(gradient.hidden_weights)))
        target=mlp_forward(BOOLEAN_INPUTS,book_initial())
        already=mlp_train(BOOLEAN_INPUTS,target,initial=book_initial())
        self.assertEqual(already.epochs,0);self.assertEqual(already.status,'converged')
        calls=[lambda:perceptron_train(BOOLEAN_INPUTS,[0,1]),lambda:perceptron_train(BOOLEAN_INPUTS,[0,0,0,2]),
            lambda:mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,learning_rate=0),
            lambda:mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,learning_rate=True),
            lambda:mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,mode='legacy'),
            lambda:mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,tolerance=0),
            lambda:mlp_forward([[np.nan,0]],book_initial()),
            lambda:mlp_loss_and_gradients(BOOLEAN_INPUTS,np.full((4,1),2),book_initial()),
            lambda:mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,initial=book_initial(),hidden_width=3)]
        extreme=MLPParameters(np.zeros((2,1)),np.zeros(2),np.array([[1e308,-1e308]]),np.zeros(1))
        with self.assertRaisesRegex(ValueError,'Nonfinite MLP'):
            mlp_loss_and_gradients([[1e308]],[[0]],extreme)
        for call in calls:
            with self.assertRaises(ValueError):call()

if __name__=='__main__':unittest.main()
