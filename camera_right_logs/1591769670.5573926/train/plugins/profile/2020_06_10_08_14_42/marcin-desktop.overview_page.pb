�	�̕A�%1@�̕A�%1@!�̕A�%1@	~���t�?~���t�?!~���t�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�̕A�%1@�#��?A��Xm�1@Y'jin���?*	��Q�6S@2F
Iterator::Model_z�sѐ�?!;4@s�QF@)6?�Ң>�?1ۮ��	@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatUK:��l�?!Z+��_i7@)�1���?1q�c�5@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate��`ũ֒?!g�t���7@)4�9A��?1��W ~!3@:Preprocessing2S
Iterator::Model::ParallelMapk�) ƃ?!{�t )@)k�) ƃ?1{�t )@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��k�ȥ?!�˿��K@)a��L�o?1�ģ%@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice9% &�Bn?!s �9@)9% &�Bn?1s �9@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�)U��-�?!6i�Ff�:@)����b?1s�%3��@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�����W?!�N���/�?)�����W?1�N���/�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�#��?�#��?!�#��?      ��!       "      ��!       *      ��!       2	��Xm�1@��Xm�1@!��Xm�1@:      ��!       B      ��!       J	'jin���?'jin���?!'jin���?R      ��!       Z	'jin���?'jin���?!'jin���?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 