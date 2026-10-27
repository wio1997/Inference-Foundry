import copy,json,unittest
from audit_raw_identity import join_marker

class IdentityTests(unittest.TestCase):
    def fixture(self):
        tag='pid123-compile1:layer00:pre'
        annotation={'q27_marker_id':tag,'q27_marker_schema':1}
        q={'marker_id':tag,'capture_id':'capture2','graph_python_id':987,'device_id':0,
           'annotation_json':json.dumps(annotation),'query_phase':'post_capture_before_graph_reset',
           'capture_status':1,'model_ri_nonnull':True,'extend_info_rc':0,'task_after_rc':0,
           'model_ri_get_id_rc':0,'get_streams_rc':0,'capture_rc':0,'callback_rc':0,
           'annotation_task_id_unchanged':True,'task_annotation_status':'annotated_exact_last_task_candidate',
           'task_id_before_annotation':5,'task_id_after_annotation':5,'physical_stream_id':39,
           'model_ri_id':49,'model_streams':[{'stream_get_id_rc':0,'model_stream_id':12},
                                          {'stream_get_id_rc':0,'model_stream_id':13}],
           'ri_bridge_status':'unknown_multiple_model_streams'}
        marker={'model_ri_query':q,'marker_id':tag,'capture_id':'capture2','graph_python_id':987,
                'device':0,'pid':123,'rank_id':0,'callback_capture_status':1,'callback_nonnull':True,
                'callback_return_codes':{'callback_rc':0,'task_rc':0,'stream_rc':0,'capture_rc':0},
                'task_id':5,'acl_stream_id':39,'layer_idx':0,'marker_side':'pre'}
        node={'pid':'123 aclGraph','args':{'ExtendInfo':json.dumps(annotation),
              'Task Type':'EVENT_RECORD','Model Id':49,'Stream Id':13,'Task Id':5}}
        return marker,[node]

    def test_multistream_tag_joins_without_stream_order_or_physical_alias(self):
        marker,nodes=self.fixture()
        result=join_marker(marker,nodes)
        self.assertEqual(result['raw_task_key'],[49,13,5])
        self.assertEqual(result['RI_bridge_status'],'unknown_multiple_model_streams')
        marker['model_ri_query']['model_streams'].reverse()
        self.assertEqual(join_marker(marker,nodes)['raw_task_key'],[49,13,5])

    def test_missing_or_duplicate_annotation_never_guesses(self):
        marker,nodes=self.fixture()
        for bad in [[],nodes+copy.deepcopy(nodes)]:
            with self.assertRaises(AssertionError):join_marker(marker,bad)

    def test_wrong_stream_task_graph_query_phase_or_api_rc_rejected(self):
        for field,value in [('Stream Id',39),('Task Id',6),('Model Id',50),('Task Type','KERNEL_AIVEC')]:
            marker,nodes=self.fixture();nodes[0]['args'][field]=value
            with self.assertRaises(AssertionError):join_marker(marker,nodes)
        for field,value in [('graph_python_id',988),('query_phase','after_reset'),('extend_info_rc',1)]:
            marker,nodes=self.fixture();marker['model_ri_query'][field]=value
            with self.assertRaises(AssertionError):join_marker(marker,nodes)

if __name__=='__main__':unittest.main()
