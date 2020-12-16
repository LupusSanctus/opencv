import os

import cv2
import tensorflow as tf
import tf2onnx
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def get_tf_model_proto(tf_model, pb_model_path, pb_model_name):
    # pb_model_name = "efficientNetB0_frozen_3.pb"
    tf_model_graph = tf.function(lambda x: tf_model(x))

    tf_model_graph = tf_model_graph.get_concrete_function(
        tf.TensorSpec(tf_model.inputs[0].shape, tf_model.inputs[0].dtype))

    frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
    frozen_tf_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                      logdir=pb_model_path,
                      name=pb_model_name,
                      as_text=False)

    return os.path.join(pb_model_path, pb_model_name)


def main():
    # modify in accordance with Note section
    is_reproduce_1st_point = True
    is_reproduce_2nd_point = False
    is_reproduce_fix = False

    dir_for_test_graphs = "effnet_models"

    opt_frozen_tf_model_name_imp = "opt_frozen_inf_graph_improved.pb"

    model = EfficientNetB0(include_top=True, weights='imagenet', classes=1000)
    model_activation = EfficientNetB0(include_top=True, weights='imagenet', activation="relu", classes=1000)

    if is_reproduce_1st_point:
        frozen_model_name = "efficientNetB0_frozen.pb"
        model_path1 = get_tf_model_proto(model, dir_for_test_graphs, frozen_model_name)

        # error caused by inputs: functional_1/efficientnetb0/normalization/Reshape/ReadVariableOp
        cv2.dnn.readNetFromTensorflow(model_path1)
        print("=== Passed #1 ===")

    if is_reproduce_2nd_point:
        """
        Note: run graph transformation for below line reproduction:
            bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=<your_path>/effnet_models/efficientNetB0_frozen.pb \
            --out_graph=<your_path>/effnet_models/opt_frozen_inf_graph.pb \
            --inputs="x" --outputs="Identity" \
            --transforms='remove_nodes(op=Identity) strip_unused_nodes fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

            or comment appropriate lines.
        """
        opt_frozen_tf_model_name = "opt_frozen_inf_graph.pb"
        frozen_opt_model_path = os.path.join(dir_for_test_graphs, opt_frozen_tf_model_name)
        if os.path.exists(os.path.join(os.getcwd(), frozen_opt_model_path)):
            # without ReadVariableOp, but contains IdentityN =>
            # Can't create layer "efficientnetb0/stem_activation/IdentityN" of type "IdentityN" in function 'getLayerInstance'
            cv2.dnn.readNetFromTensorflow(frozen_opt_model_path)
            print("=== Passed #2 ===")
        else:
            print("=== Run transform_graph tool in accordance with instructions")

    if is_reproduce_fix:
        # freeze => transform
        frozen_model_name_imp = "efficientNetB0_frozen_improved.pb"
        opt_frozen_model_name_imp = "opt_frozen_inf_graph_improved.pb"
        frozen_opt_model_path_imp = os.path.join(dir_for_test_graphs, opt_frozen_model_name_imp)

        if not os.path.exists(os.path.join(os.getcwd(), frozen_opt_model_path_imp)):
            get_tf_model_proto(model_activation, dir_for_test_graphs, frozen_model_name_imp)

        """
        Note: run graph transformation for below line reproduction:
            bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=<your_path>/effnet_models/efficientNetB0_frozen_improved.pb \
            --out_graph=<your_path>/effnet_models/opt_frozen_inf_graph_improved.pb \
            --inputs="x" --outputs="Identity" \
            --transforms='remove_nodes(op=Identity) strip_unused_nodes fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

            or comment appropriate lines.
        """

        if os.path.exists(os.path.join(os.getcwd(), frozen_opt_model_path_imp)):
            opencv_net3 = cv2.dnn.readNetFromTensorflow(frozen_opt_model_path_imp)
            print(opencv_net3.getLayerNames())
            print("=== Passed #3 ===")
        else:
            print("=== Run transform_graph tool in accordance with instructions")


if __name__ == "__main__":
    main()
