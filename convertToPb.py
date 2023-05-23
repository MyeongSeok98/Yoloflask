import tensorflow as tf
from tensorflow.python.framework import graph_util

# 파일 경로 설정
index_file = './model/ckpt-51.index'
checkpoint_file = './model/checkpoint'
data_file = './model/ckpt-51.data-00000-of-00001'
config_file = './model/pipeline.config'
output_pb_file = './model/model.pb'

# eager execution 비활성화
tf.compat.v1.disable_eager_execution()

# 모델 불러오기
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    # 가중치 복원하기
    saver = tf.compat.v1.train.import_meta_graph(index_file, clear_devices=True)
    saver.restore(sess, checkpoint_file)

    # GraphDef 추출하기
    graph = tf.compat.v1.get_default_graph()
    graph_def = graph.as_graph_def()

    # GraphDef 최적화하기
    output_node_names = ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)

    # Protobuf(pb) 파일로 저장하기
    with tf.io.gfile.GFile(output_pb_file, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

print('Model converted to', output_pb_file)