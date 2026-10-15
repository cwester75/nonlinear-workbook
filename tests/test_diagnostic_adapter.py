import unittest
import numpy as np
from nonlinear_workbook.diagnostic_adapter import WindowConfig,diagnose_prefix

class AdapterTests(unittest.TestCase):
    def config(self,**kwargs):
        return WindowConfig(**dict(dict(window=256,sample_interval=.1,time_unit='second',value_unit='normalized amplitude',epsilon=.3),**kwargs))

    def test_causal_provenance_and_units(self):
        x=np.sin(np.arange(512)*.1); kw=dict(end_index=256,config=self.config(),source_id='synthetic-sine',workbook_commit='a'*40)
        before=diagnose_prefix(x,**kw); x[256:]=np.nan
        self.assertEqual(before,diagnose_prefix(x,**kw))
        self.assertEqual(before['feature_units']['divergence'],'1/second')
        self.assertEqual((before['start_index'],before['end_index']),(0,256))
        shifted=diagnose_prefix(np.sin(np.arange(512)*.1),**dict(kw,end_index=512))
        self.assertEqual(shifted['start_index'],256)

    def test_unavailable_and_constant_windows(self):
        kw=dict(config=self.config(),source_id='fixture',workbook_commit='a'*40)
        result=diagnose_prefix(np.ones(256),end_index=256,**kw)
        self.assertEqual(result['status'],'partial'); self.assertIsNone(result['features']['hurst'])
        self.assertIsNone(result['features']['divergence']); self.assertEqual(result['features']['lz'],0)
        self.assertEqual(diagnose_prefix(np.ones(256),end_index=32,**kw)['status'],'insufficient_window')
        self.assertEqual(diagnose_prefix(np.full(256,np.nan),end_index=256,**kw)['status'],'invalid_window')

    def test_time_scaling_and_seeded_repeat(self):
        x=np.random.default_rng(7).normal(size=256); kw=dict(end_index=256,source_id='seed7',workbook_commit='a'*40)
        a=diagnose_prefix(x,config=self.config(),**kw); b=diagnose_prefix(x,config=self.config(sample_interval=.2),**kw)
        self.assertEqual(a,diagnose_prefix(x,config=self.config(),**kw))
        self.assertIsNotNone(a['features']['divergence'])
        self.assertAlmostEqual(a['features']['divergence'],2*b['features']['divergence'])

    def test_invalid_configuration(self):
        for override in (dict(window=32),dict(time_unit=''),dict(fit_stop=20),dict(theiler=300),dict(epsilon=0)):
            with self.assertRaises(ValueError): self.config(**override)
        with self.assertRaises(ValueError): diagnose_prefix([1,2],end_index=3,config=self.config(),source_id='x',workbook_commit='a'*40)

if __name__=='__main__': unittest.main()
