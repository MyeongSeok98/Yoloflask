import tensorflow as tf

# 그래프를 로드합니다.
graph = tf.compat.v1.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # .meta 파일을 로드하여 그래프 구조를 복원합니다.
    saver = tf.compat.v1.train.import_meta_graph('./model/model.ckpt-200000.meta')

    # 그래프 구조에서 출력 노드의 이름을 찾습니다.
    output_node_names = []
    for op in graph.get_operations():
        if len(op.outputs) == 1 and op.outputs[0].dtype == tf.float32:
            output_node_names.append(op.name)

    print(output_node_names)