# SPDX-License-Identifier: Apache-2.0
"""Dependency declarations and exact expert byte accounting without devices."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from deepinfer.hugging_face.glm5_next_native import KDA_PROFILE,KDA_W8A8_BINDINGS,NativeDependencyDeclaration,plan_expert_residency


def manifest(ids=(1,3),shared=True):
    def spec(i):
        prefix='shared' if i is None else f'expert.{i}'
        return NS(global_id=i,prefixes=tuple(prefix+'.'+p for p in ('gate','up','down')),shapes=((5,3),(5,3),(3,5)))
    return NS(num_experts=7,hidden_size=3,routed=tuple(spec(i) for i in ids),shared=spec(None) if shared else None)


class NativePreparationTests(unittest.TestCase):
    def test_matching_declaration_is_not_native_or_model_readiness(self):
        result=NativeDependencyDeclaration('aarch64',KDA_PROFILE,KDA_W8A8_BINDINGS).require_matching_declaration()
        self.assertTrue(result['declaration_matches']);self.assertFalse(result['native_execution_validated']);self.assertFalse(result['model_ready'])

    def test_versions_architecture_missing_exports_fail_closed(self):
        base=NativeDependencyDeclaration('aarch64',KDA_PROFILE,KDA_W8A8_BINDINGS)
        for d in (replace(base,architecture='x86_64'),replace(base,packages=tuple((n,'2.13.0+cpu') if n=='torch' else (n,v) for n,v in KDA_PROFILE)),replace(base,bindings=frozenset())):
            with self.assertRaises(ValueError):d.require_matching_declaration()
        # A matching distribution tuple never implies the exports exist.
        self.assertEqual(len(replace(base,bindings=frozenset()).report()['missing_bindings']),6)

    def test_duplicate_or_incomplete_package_inventory_rejected(self):
        for packages in (KDA_PROFILE[:-1],KDA_PROFILE+(KDA_PROFILE[0],),list(KDA_PROFILE)):
            with self.assertRaises(ValueError):NativeDependencyDeclaration('aarch64',packages,KDA_W8A8_BINDINGS).report()
        with self.assertRaises(ValueError):NativeDependencyDeclaration('aarch64',KDA_PROFILE,set(KDA_W8A8_BINDINGS)).report()

    def test_metadata_does_not_import_or_infer_native_bindings(self):
        with patch('importlib.metadata.version',side_effect=lambda n:dict(KDA_PROFILE)[n]),patch('platform.machine',return_value='aarch64'):
            d=NativeDependencyDeclaration.from_metadata()
        self.assertEqual(d.packages,KDA_PROFILE);self.assertEqual(d.bindings,frozenset());self.assertFalse(d.report()['declaration_matches'])

    def test_exact_weight_sidecar_and_resident_arithmetic(self):
        # Each expert has45 I8 elements and13 channel rows, hence149/71 bytes.
        p=plan_expert_residency(manifest(),max_checkpoint_bytes=447,max_resident_bytes=213)
        self.assertEqual((p.checkpoint_payload_bytes,p.resident_logical_bytes,p.largest_weight_bytes),(447,213,15))
        self.assertEqual((p.routed_experts,p.includes_shared),(2,True));self.assertIsNone(p.to_dict()['physical_peak_bytes'])

    def test_each_budget_is_inclusive_and_rejects_without_truncating(self):
        m=manifest()
        for a,b in ((446,213),(447,212),(0,213),(447,True)):
            with self.assertRaises(ValueError):plan_expert_residency(m,max_checkpoint_bytes=a,max_resident_bytes=b)
        self.assertEqual([s.global_id for s in m.routed],[1,3]);self.assertIsNotNone(m.shared)

    def test_empty_and_shared_only_banks(self):
        for shared,expected in ((False,0),(True,149)):
            p=plan_expert_residency(manifest((),shared),max_checkpoint_bytes=149,max_resident_bytes=71)
            self.assertEqual(p.checkpoint_payload_bytes,expected);self.assertEqual(p.routed_experts,0)

    def test_geometry_namespace_and_duplicate_matrix_rejected(self):
        cases=[manifest((3,1)),manifest((1,1)),manifest((-1,)),manifest((7,))]
        m=manifest();m.routed[0].shapes=((5,3),(5,3),(3,6));cases.append(m)
        m=manifest();m.routed[0].prefixes=m.routed[1].prefixes;cases.append(m)
        m=manifest();m.shared.global_id=2;cases.append(m)
        m=manifest();m.shared.shapes=((6,3),(6,3),(3,6));cases.append(m)
        for m in cases:
            with self.assertRaises(ValueError):plan_expert_residency(m,max_checkpoint_bytes=9999,max_resident_bytes=9999)

    def test_native_loader_budget_precedes_payload_and_native_import(self):
        # Substitute device classification/metadata only; no device is created.
        import deepinfer.model_executor.models.glm5_next.expert_bank as bank
        class Reader:
            def linear_payload(self,*a):raise AssertionError('payload read before budget rejection')
        with patch.object(bank,'torch',NS(device=lambda _:NS(type='npu'))),patch.object(bank,'selected_expert_manifest',return_value=manifest()):
            with self.assertRaisesRegex(ValueError,'checkpoint expert payload'):
                bank.load_selected_expert_bank(Reader(),3,[1,3],'classification-only',max_checkpoint_bytes=1)
        # Existing CPU guard still rejects before reading any checkpoint.
        with self.assertRaisesRegex(ValueError,'requires authorized NPU'):
            bank.load_selected_expert_bank(Reader(),3,[1,3],'cpu')


if __name__=='__main__':unittest.main(verbosity=2)
