??	
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878ǰ
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
variational_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*(
shared_namevariational_mean/kernel
?
+variational_mean/kernel/Read/ReadVariableOpReadVariableOpvariational_mean/kernel*
_output_shapes
:	?d*
dtype0
?
variational_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_namevariational_mean/bias
{
)variational_mean/bias/Read/ReadVariableOpReadVariableOpvariational_mean/bias*
_output_shapes
:d*
dtype0
?
variational_log_variance/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*0
shared_name!variational_log_variance/kernel
?
3variational_log_variance/kernel/Read/ReadVariableOpReadVariableOpvariational_log_variance/kernel*
_output_shapes
:	?d*
dtype0
?
variational_log_variance/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namevariational_log_variance/bias
?
1variational_log_variance/bias/Read/ReadVariableOpReadVariableOpvariational_log_variance/bias*
_output_shapes
:d*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
R
+regularization_losses
,trainable_variables
-	variables
.	keras_api
R
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
R
?regularization_losses
@trainable_variables
A	variables
B	keras_api
 
F
0
1
2
3
%4
&5
36
47
98
:9
F
0
1
2
3
%4
&5
36
47
98
:9
?
Clayer_metrics
regularization_losses
Dmetrics

Elayers
	variables
Fnon_trainable_variables
trainable_variables
Glayer_regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Hlayer_metrics
regularization_losses

Ilayers
trainable_variables
	variables
Jnon_trainable_variables
Kmetrics
Llayer_regularization_losses
 
 
 
?
Mlayer_metrics
regularization_losses

Nlayers
trainable_variables
	variables
Onon_trainable_variables
Pmetrics
Qlayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Rlayer_metrics
regularization_losses

Slayers
trainable_variables
	variables
Tnon_trainable_variables
Umetrics
Vlayer_regularization_losses
 
 
 
?
Wlayer_metrics
!regularization_losses

Xlayers
"trainable_variables
#	variables
Ynon_trainable_variables
Zmetrics
[layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
?
\layer_metrics
'regularization_losses

]layers
(trainable_variables
)	variables
^non_trainable_variables
_metrics
`layer_regularization_losses
 
 
 
?
alayer_metrics
+regularization_losses

blayers
,trainable_variables
-	variables
cnon_trainable_variables
dmetrics
elayer_regularization_losses
 
 
 
?
flayer_metrics
/regularization_losses

glayers
0trainable_variables
1	variables
hnon_trainable_variables
imetrics
jlayer_regularization_losses
ca
VARIABLE_VALUEvariational_mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEvariational_mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
klayer_metrics
5regularization_losses

llayers
6trainable_variables
7	variables
mnon_trainable_variables
nmetrics
olayer_regularization_losses
ki
VARIABLE_VALUEvariational_log_variance/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEvariational_log_variance/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
?
player_metrics
;regularization_losses

qlayers
<trainable_variables
=	variables
rnon_trainable_variables
smetrics
tlayer_regularization_losses
 
 
 
?
ulayer_metrics
?regularization_losses

vlayers
@trainable_variables
A	variables
wnon_trainable_variables
xmetrics
ylayer_regularization_losses
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????--*
dtype0*$
shape:?????????--
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasvariational_mean/kernelvariational_mean/biasvariational_log_variance/kernelvariational_log_variance/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_10626
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp+variational_mean/kernel/Read/ReadVariableOp)variational_mean/bias/Read/ReadVariableOp3variational_log_variance/kernel/Read/ReadVariableOp1variational_log_variance/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_10998
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasvariational_mean/kernelvariational_mean/biasvariational_log_variance/kernelvariational_log_variance/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_11038??
?
?
8__inference_variational_log_variance_layer_call_fn_10881

inputs#
variational_log_variance_kernel!
variational_log_variance_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_104262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10286

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?0
?
G__inference_functional_1_layer_call_and_return_conditional_losses_10506
input_1
conv2d_conv2d_kernel
conv2d_conv2d_bias
conv2d_1_conv2d_1_kernel
conv2d_1_conv2d_1_bias
conv2d_2_conv2d_2_kernel
conv2d_2_conv2d_2_bias,
(variational_mean_variational_mean_kernel*
&variational_mean_variational_mean_bias<
8variational_log_variance_variational_log_variance_kernel:
6variational_log_variance_variational_log_variance_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?0variational_log_variance/StatefulPartitionedCall?(variational_mean/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103042 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_102522
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103332"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_102692!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103622"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_102862!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_103862
flatten/PartitionedCall?
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_variational_mean_layer_call_and_return_conditional_losses_104042*
(variational_mean/StatefulPartitionedCall?
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_1042622
0variational_log_variance/StatefulPartitionedCall?
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_104642 
lambda/StatefulPartitionedCall?
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:X T
/
_output_shapes
:?????????--
!
_user_specified_name	input_1
?J
?
G__inference_functional_1_layer_call_and_return_conditional_losses_10689

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel-
)conv2d_biasadd_readvariableop_conv2d_bias2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel1
-conv2d_1_biasadd_readvariableop_conv2d_1_bias2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernel1
-conv2d_2_biasadd_readvariableop_conv2d_2_biasB
>variational_mean_matmul_readvariableop_variational_mean_kernelA
=variational_mean_biasadd_readvariableop_variational_mean_biasR
Nvariational_log_variance_matmul_readvariableop_variational_log_variance_kernelQ
Mvariational_log_variance_biasadd_readvariableop_variational_log_variance_bias
identity??
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????))@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????		@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
&variational_mean/MatMul/ReadVariableOpReadVariableOp>variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	?d*
dtype02(
&variational_mean/MatMul/ReadVariableOp?
variational_mean/MatMulMatMulflatten/Reshape:output:0.variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
variational_mean/MatMul?
'variational_mean/BiasAdd/ReadVariableOpReadVariableOp=variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02)
'variational_mean/BiasAdd/ReadVariableOp?
variational_mean/BiasAddBiasAdd!variational_mean/MatMul:product:0/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
variational_mean/BiasAdd?
.variational_log_variance/MatMul/ReadVariableOpReadVariableOpNvariational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	?d*
dtype020
.variational_log_variance/MatMul/ReadVariableOp?
variational_log_variance/MatMulMatMulflatten/Reshape:output:06variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
variational_log_variance/MatMul?
/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpMvariational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype021
/variational_log_variance/BiasAdd/ReadVariableOp?
 variational_log_variance/BiasAddBiasAdd)variational_log_variance/MatMul:product:07variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 variational_log_variance/BiasAddu
lambda/ShapeShape)variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
lambda/Shape?
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lambda/strided_slice/stack?
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_1?
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_2?
lambda/strided_sliceStridedSlicelambda/Shape:output:0#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lambda/strided_slicey
lambda/Shape_1Shape)variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
lambda/Shape_1?
lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice_1/stack?
lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_1?
lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_2?
lambda/strided_slice_1StridedSlicelambda/Shape_1:output:0%lambda/strided_slice_1/stack:output:0'lambda/strided_slice_1/stack_1:output:0'lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lambda/strided_slice_1?
lambda/random_normal/shapePacklambda/strided_slice:output:0lambda/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
lambda/random_normal/shape{
lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/random_normal/mean
lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lambda/random_normal/stddev?
)lambda/random_normal/RandomStandardNormalRandomStandardNormal#lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2+
)lambda/random_normal/RandomStandardNormal?
lambda/random_normal/mulMul2lambda/random_normal/RandomStandardNormal:output:0$lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
lambda/random_normal/mul?
lambda/random_normalAddlambda/random_normal/mul:z:0"lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
lambda/random_normala
lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda/mul/x?

lambda/mulMullambda/mul/x:output:0)variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2

lambda/mula

lambda/ExpExplambda/mul:z:0*
T0*'
_output_shapes
:?????????d2

lambda/Exp
lambda/mul_1Mullambda/Exp:y:0lambda/random_normal:z:0*
T0*'
_output_shapes
:?????????d2
lambda/mul_1?

lambda/addAddV2!variational_mean/BiasAdd:output:0lambda/mul_1:z:0*
T0*'
_output_shapes
:?????????d2

lambda/addb
IdentityIdentitylambda/add:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--:::::::::::W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?0
?
G__inference_functional_1_layer_call_and_return_conditional_losses_10596

inputs
conv2d_conv2d_kernel
conv2d_conv2d_bias
conv2d_1_conv2d_1_kernel
conv2d_1_conv2d_1_bias
conv2d_2_conv2d_2_kernel
conv2d_2_conv2d_2_bias,
(variational_mean_variational_mean_kernel*
&variational_mean_variational_mean_bias<
8variational_log_variance_variational_log_variance_kernel:
6variational_log_variance_variational_log_variance_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?0variational_log_variance/StatefulPartitionedCall?(variational_mean/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103042 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_102522
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103332"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_102692!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103622"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_102862!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_103862
flatten/PartitionedCall?
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_variational_mean_layer_call_and_return_conditional_losses_104042*
(variational_mean/StatefulPartitionedCall?
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_1042622
0variational_log_variance/StatefulPartitionedCall?
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_104902 
lambda/StatefulPartitionedCall?
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?
p
A__inference_lambda_layer_call_and_return_conditional_losses_10933
inputs_0
inputs_1
identity?F
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_1*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2??B2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:?????????d2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????d2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????d2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?#
?
__inference__traced_save_10998
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop6
2savev2_variational_mean_kernel_read_readvariableop4
0savev2_variational_mean_bias_read_readvariableop>
:savev2_variational_log_variance_kernel_read_readvariableop<
8savev2_variational_log_variance_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_626e125cb450496ba4d8e04f8d420056/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop2savev2_variational_mean_kernel_read_readvariableop0savev2_variational_mean_bias_read_readvariableop:savev2_variational_log_variance_kernel_read_readvariableop8savev2_variational_log_variance_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapesp
n: :@:@:@@:@:@ : :	?d:d:	?d:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :%!

_output_shapes
:	?d: 

_output_shapes
:d:%	!

_output_shapes
:	?d: 


_output_shapes
:d:

_output_shapes
: 
?
o
&__inference_lambda_layer_call_fn_10945
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_104902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?	
?
A__inference_conv2d_layer_call_and_return_conditional_losses_10304

inputs'
#conv2d_readvariableop_conv2d_kernel&
"biasadd_readvariableop_conv2d_bias
identity??
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????))@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????))@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????--:::W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?
n
A__inference_lambda_layer_call_and_return_conditional_losses_10490

inputs
inputs_1
identity?F
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_1*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:?????????d2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????d2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????d2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_10874

inputs9
5matmul_readvariableop_variational_log_variance_kernel8
4biasadd_readvariableop_variational_log_variance_bias
identity??
MatMul/ReadVariableOpReadVariableOp5matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp4biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_10386

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?J
?
G__inference_functional_1_layer_call_and_return_conditional_losses_10752

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel-
)conv2d_biasadd_readvariableop_conv2d_bias2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel1
-conv2d_1_biasadd_readvariableop_conv2d_1_bias2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernel1
-conv2d_2_biasadd_readvariableop_conv2d_2_biasB
>variational_mean_matmul_readvariableop_variational_mean_kernelA
=variational_mean_biasadd_readvariableop_variational_mean_biasR
Nvariational_log_variance_matmul_readvariableop_variational_log_variance_kernelQ
Mvariational_log_variance_biasadd_readvariableop_variational_log_variance_bias
identity??
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????))@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????		@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
&variational_mean/MatMul/ReadVariableOpReadVariableOp>variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	?d*
dtype02(
&variational_mean/MatMul/ReadVariableOp?
variational_mean/MatMulMatMulflatten/Reshape:output:0.variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
variational_mean/MatMul?
'variational_mean/BiasAdd/ReadVariableOpReadVariableOp=variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02)
'variational_mean/BiasAdd/ReadVariableOp?
variational_mean/BiasAddBiasAdd!variational_mean/MatMul:product:0/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
variational_mean/BiasAdd?
.variational_log_variance/MatMul/ReadVariableOpReadVariableOpNvariational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	?d*
dtype020
.variational_log_variance/MatMul/ReadVariableOp?
variational_log_variance/MatMulMatMulflatten/Reshape:output:06variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
variational_log_variance/MatMul?
/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpMvariational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype021
/variational_log_variance/BiasAdd/ReadVariableOp?
 variational_log_variance/BiasAddBiasAdd)variational_log_variance/MatMul:product:07variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 variational_log_variance/BiasAddu
lambda/ShapeShape)variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
lambda/Shape?
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lambda/strided_slice/stack?
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_1?
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_2?
lambda/strided_sliceStridedSlicelambda/Shape:output:0#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lambda/strided_slicey
lambda/Shape_1Shape)variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
lambda/Shape_1?
lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice_1/stack?
lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_1?
lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_2?
lambda/strided_slice_1StridedSlicelambda/Shape_1:output:0%lambda/strided_slice_1/stack:output:0'lambda/strided_slice_1/stack_1:output:0'lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lambda/strided_slice_1?
lambda/random_normal/shapePacklambda/strided_slice:output:0lambda/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
lambda/random_normal/shape{
lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/random_normal/mean
lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lambda/random_normal/stddev?
)lambda/random_normal/RandomStandardNormalRandomStandardNormal#lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2+
)lambda/random_normal/RandomStandardNormal?
lambda/random_normal/mulMul2lambda/random_normal/RandomStandardNormal:output:0$lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
lambda/random_normal/mul?
lambda/random_normalAddlambda/random_normal/mul:z:0"lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
lambda/random_normala
lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda/mul/x?

lambda/mulMullambda/mul/x:output:0)variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2

lambda/mula

lambda/ExpExplambda/mul:z:0*
T0*'
_output_shapes
:?????????d2

lambda/Exp
lambda/mul_1Mullambda/Exp:y:0lambda/random_normal:z:0*
T0*'
_output_shapes
:?????????d2
lambda/mul_1?

lambda/addAddV2!variational_mean/BiasAdd:output:0lambda/mul_1:z:0*
T0*'
_output_shapes
:?????????d2

lambda/addb
IdentityIdentitylambda/add:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--:::::::::::W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?
o
&__inference_lambda_layer_call_fn_10939
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_104642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
I
-__inference_max_pooling2d_layer_call_fn_10255

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_102522
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
n
A__inference_lambda_layer_call_and_return_conditional_losses_10464

inputs
inputs_1
identity?F
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_1*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:?????????d2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????d2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????d2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_10261

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_10426

inputs9
5matmul_readvariableop_variational_log_variance_kernel8
4biasadd_readvariableop_variational_log_variance_bias
identity??
MatMul/ReadVariableOpReadVariableOp5matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp4biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
p
A__inference_lambda_layer_call_and_return_conditional_losses_10907
inputs_0
inputs_1
identity?F
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_1*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2?ȕ2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:?????????d2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????d2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????d2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_10269

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10829

inputs)
%conv2d_readvariableop_conv2d_2_kernel(
$biasadd_readvariableop_conv2d_2_bias
identity??
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????		@:::W S
/
_output_shapes
:?????????		@
 
_user_specified_nameinputs
?0
?
G__inference_functional_1_layer_call_and_return_conditional_losses_10530
input_1
conv2d_conv2d_kernel
conv2d_conv2d_bias
conv2d_1_conv2d_1_kernel
conv2d_1_conv2d_1_bias
conv2d_2_conv2d_2_kernel
conv2d_2_conv2d_2_bias,
(variational_mean_variational_mean_kernel*
&variational_mean_variational_mean_bias<
8variational_log_variance_variational_log_variance_kernel:
6variational_log_variance_variational_log_variance_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?0variational_log_variance/StatefulPartitionedCall?(variational_mean/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103042 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_102522
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103332"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_102692!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103622"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_102862!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_103862
flatten/PartitionedCall?
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_variational_mean_layer_call_and_return_conditional_losses_104042*
(variational_mean/StatefulPartitionedCall?
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_1042622
0variational_log_variance/StatefulPartitionedCall?
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_104902 
lambda/StatefulPartitionedCall?
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:X T
/
_output_shapes
:?????????--
!
_user_specified_name	input_1
?	
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10811

inputs)
%conv2d_readvariableop_conv2d_1_kernel(
$biasadd_readvariableop_conv2d_1_bias
identity??
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_2_layer_call_fn_10836

inputs
conv2d_2_kernel
conv2d_2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernelconv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????		@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????		@
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_10847

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_103862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10252

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?0
?
G__inference_functional_1_layer_call_and_return_conditional_losses_10557

inputs
conv2d_conv2d_kernel
conv2d_conv2d_bias
conv2d_1_conv2d_1_kernel
conv2d_1_conv2d_1_bias
conv2d_2_conv2d_2_kernel
conv2d_2_conv2d_2_bias,
(variational_mean_variational_mean_kernel*
&variational_mean_variational_mean_bias<
8variational_log_variance_variational_log_variance_kernel:
6variational_log_variance_variational_log_variance_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?0variational_log_variance/StatefulPartitionedCall?(variational_mean/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103042 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_102522
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103332"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_102692!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_103622"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_102862!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_103862
flatten/PartitionedCall?
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_variational_mean_layer_call_and_return_conditional_losses_104042*
(variational_mean/StatefulPartitionedCall?
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_1042622
0variational_log_variance/StatefulPartitionedCall?
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_104642 
lambda/StatefulPartitionedCall?
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?

?
,__inference_functional_1_layer_call_fn_10609
input_1
conv2d_kernel
conv2d_bias
conv2d_1_kernel
conv2d_1_bias
conv2d_2_kernel
conv2d_2_bias
variational_mean_kernel
variational_mean_bias#
variational_log_variance_kernel!
variational_log_variance_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_105962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????--
!
_user_specified_name	input_1
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10244

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
K__inference_variational_mean_layer_call_and_return_conditional_losses_10404

inputs1
-matmul_readvariableop_variational_mean_kernel0
,biasadd_readvariableop_variational_mean_bias
identity??
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
,__inference_functional_1_layer_call_fn_10767

inputs
conv2d_kernel
conv2d_bias
conv2d_1_kernel
conv2d_1_bias
conv2d_2_kernel
conv2d_2_bias
variational_mean_kernel
variational_mean_bias#
variational_log_variance_kernel!
variational_log_variance_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_105572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_10818

inputs
conv2d_1_kernel
conv2d_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernelconv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_103332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_2_layer_call_fn_10289

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_102862
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
,__inference_functional_1_layer_call_fn_10782

inputs
conv2d_kernel
conv2d_bias
conv2d_1_kernel
conv2d_1_bias
conv2d_2_kernel
conv2d_2_bias
variational_mean_kernel
variational_mean_bias#
variational_log_variance_kernel!
variational_log_variance_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_105962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?	
?
A__inference_conv2d_layer_call_and_return_conditional_losses_10793

inputs'
#conv2d_readvariableop_conv2d_kernel&
"biasadd_readvariableop_conv2d_bias
identity??
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????))@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????))@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????--:::W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10362

inputs)
%conv2d_readvariableop_conv2d_2_kernel(
$biasadd_readvariableop_conv2d_2_bias
identity??
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????		@:::W S
/
_output_shapes
:?????????		@
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_10626
input_1
conv2d_kernel
conv2d_bias
conv2d_1_kernel
conv2d_1_bias
conv2d_2_kernel
conv2d_2_bias
variational_mean_kernel
variational_mean_bias#
variational_log_variance_kernel!
variational_log_variance_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_102382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????--
!
_user_specified_name	input_1
?[
?
 __inference__wrapped_model_10238
input_1;
7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel:
6functional_1_conv2d_biasadd_readvariableop_conv2d_bias?
;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel>
:functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias?
;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel>
:functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_biasO
Kfunctional_1_variational_mean_matmul_readvariableop_variational_mean_kernelN
Jfunctional_1_variational_mean_biasadd_readvariableop_variational_mean_bias_
[functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel^
Zfunctional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_bias
identity??
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOp?
functional_1/conv2d/Conv2DConv2Dinput_11functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@*
paddingVALID*
strides
2
functional_1/conv2d/Conv2D?
*functional_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp6functional_1_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02,
*functional_1/conv2d/BiasAdd/ReadVariableOp?
functional_1/conv2d/BiasAddBiasAdd#functional_1/conv2d/Conv2D:output:02functional_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????))@2
functional_1/conv2d/BiasAdd?
functional_1/conv2d/ReluRelu$functional_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????))@2
functional_1/conv2d/Relu?
"functional_1/max_pooling2d/MaxPoolMaxPool&functional_1/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2$
"functional_1/max_pooling2d/MaxPool?
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOp?
functional_1/conv2d_1/Conv2DConv2D+functional_1/max_pooling2d/MaxPool:output:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
functional_1/conv2d_1/Conv2D?
,functional_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02.
,functional_1/conv2d_1/BiasAdd/ReadVariableOp?
functional_1/conv2d_1/BiasAddBiasAdd%functional_1/conv2d_1/Conv2D:output:04functional_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
functional_1/conv2d_1/BiasAdd?
functional_1/conv2d_1/ReluRelu&functional_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
functional_1/conv2d_1/Relu?
$functional_1/max_pooling2d_1/MaxPoolMaxPool(functional_1/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????		@*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_1/MaxPool?
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOp?
functional_1/conv2d_2/Conv2DConv2D-functional_1/max_pooling2d_1/MaxPool:output:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
functional_1/conv2d_2/Conv2D?
,functional_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02.
,functional_1/conv2d_2/BiasAdd/ReadVariableOp?
functional_1/conv2d_2/BiasAddBiasAdd%functional_1/conv2d_2/Conv2D:output:04functional_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
functional_1/conv2d_2/BiasAdd?
functional_1/conv2d_2/ReluRelu&functional_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
functional_1/conv2d_2/Relu?
$functional_1/max_pooling2d_2/MaxPoolMaxPool(functional_1/conv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_2/MaxPool?
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
functional_1/flatten/Const?
functional_1/flatten/ReshapeReshape-functional_1/max_pooling2d_2/MaxPool:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
functional_1/flatten/Reshape?
3functional_1/variational_mean/MatMul/ReadVariableOpReadVariableOpKfunctional_1_variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	?d*
dtype025
3functional_1/variational_mean/MatMul/ReadVariableOp?
$functional_1/variational_mean/MatMulMatMul%functional_1/flatten/Reshape:output:0;functional_1/variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2&
$functional_1/variational_mean/MatMul?
4functional_1/variational_mean/BiasAdd/ReadVariableOpReadVariableOpJfunctional_1_variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype026
4functional_1/variational_mean/BiasAdd/ReadVariableOp?
%functional_1/variational_mean/BiasAddBiasAdd.functional_1/variational_mean/MatMul:product:0<functional_1/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2'
%functional_1/variational_mean/BiasAdd?
;functional_1/variational_log_variance/MatMul/ReadVariableOpReadVariableOp[functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	?d*
dtype02=
;functional_1/variational_log_variance/MatMul/ReadVariableOp?
,functional_1/variational_log_variance/MatMulMatMul%functional_1/flatten/Reshape:output:0Cfunctional_1/variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,functional_1/variational_log_variance/MatMul?
<functional_1/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpZfunctional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02>
<functional_1/variational_log_variance/BiasAdd/ReadVariableOp?
-functional_1/variational_log_variance/BiasAddBiasAdd6functional_1/variational_log_variance/MatMul:product:0Dfunctional_1/variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-functional_1/variational_log_variance/BiasAdd?
functional_1/lambda/ShapeShape6functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_1/lambda/Shape?
'functional_1/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'functional_1/lambda/strided_slice/stack?
)functional_1/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice/stack_1?
)functional_1/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice/stack_2?
!functional_1/lambda/strided_sliceStridedSlice"functional_1/lambda/Shape:output:00functional_1/lambda/strided_slice/stack:output:02functional_1/lambda/strided_slice/stack_1:output:02functional_1/lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!functional_1/lambda/strided_slice?
functional_1/lambda/Shape_1Shape6functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_1/lambda/Shape_1?
)functional_1/lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice_1/stack?
+functional_1/lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/lambda/strided_slice_1/stack_1?
+functional_1/lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/lambda/strided_slice_1/stack_2?
#functional_1/lambda/strided_slice_1StridedSlice$functional_1/lambda/Shape_1:output:02functional_1/lambda/strided_slice_1/stack:output:04functional_1/lambda/strided_slice_1/stack_1:output:04functional_1/lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#functional_1/lambda/strided_slice_1?
'functional_1/lambda/random_normal/shapePack*functional_1/lambda/strided_slice:output:0,functional_1/lambda/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/lambda/random_normal/shape?
&functional_1/lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&functional_1/lambda/random_normal/mean?
(functional_1/lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(functional_1/lambda/random_normal/stddev?
6functional_1/lambda/random_normal/RandomStandardNormalRandomStandardNormal0functional_1/lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???28
6functional_1/lambda/random_normal/RandomStandardNormal?
%functional_1/lambda/random_normal/mulMul?functional_1/lambda/random_normal/RandomStandardNormal:output:01functional_1/lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2'
%functional_1/lambda/random_normal/mul?
!functional_1/lambda/random_normalAdd)functional_1/lambda/random_normal/mul:z:0/functional_1/lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2#
!functional_1/lambda/random_normal{
functional_1/lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
functional_1/lambda/mul/x?
functional_1/lambda/mulMul"functional_1/lambda/mul/x:output:06functional_1/variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
functional_1/lambda/mul?
functional_1/lambda/ExpExpfunctional_1/lambda/mul:z:0*
T0*'
_output_shapes
:?????????d2
functional_1/lambda/Exp?
functional_1/lambda/mul_1Mulfunctional_1/lambda/Exp:y:0%functional_1/lambda/random_normal:z:0*
T0*'
_output_shapes
:?????????d2
functional_1/lambda/mul_1?
functional_1/lambda/addAddV2.functional_1/variational_mean/BiasAdd:output:0functional_1/lambda/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
functional_1/lambda/addo
IdentityIdentityfunctional_1/lambda/add:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--:::::::::::X T
/
_output_shapes
:?????????--
!
_user_specified_name	input_1
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_10842

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
,__inference_functional_1_layer_call_fn_10570
input_1
conv2d_kernel
conv2d_bias
conv2d_1_kernel
conv2d_1_bias
conv2d_2_kernel
conv2d_2_bias
variational_mean_kernel
variational_mean_bias#
variational_log_variance_kernel!
variational_log_variance_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_105572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????--
!
_user_specified_name	input_1
?
K
/__inference_max_pooling2d_1_layer_call_fn_10272

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_102692
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?.
?
!__inference__traced_restore_11038
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias.
*assignvariableop_6_variational_mean_kernel,
(assignvariableop_7_variational_mean_bias6
2assignvariableop_8_variational_log_variance_kernel4
0assignvariableop_9_variational_log_variance_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_variational_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp(assignvariableop_7_variational_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp2assignvariableop_8_variational_log_variance_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp0assignvariableop_9_variational_log_variance_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10278

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
0__inference_variational_mean_layer_call_fn_10864

inputs
variational_mean_kernel
variational_mean_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsvariational_mean_kernelvariational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_variational_mean_layer_call_and_return_conditional_losses_104042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_variational_mean_layer_call_and_return_conditional_losses_10857

inputs1
-matmul_readvariableop_variational_mean_kernel0
,biasadd_readvariableop_variational_mean_bias
identity??
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_10800

inputs
conv2d_kernel
conv2d_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_103042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????))@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????--::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????--
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10333

inputs)
%conv2d_readvariableop_conv2d_1_kernel(
$biasadd_readvariableop_conv2d_1_bias
identity??
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????--:
lambda0
StatefulPartitionedCall:0?????????dtensorflow/serving/predict:??
?`
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*z&call_and_return_all_conditional_losses
{_default_save_signature
|__call__"?]
_tf_keras_network?\{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_log_variance", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["variational_mean", 0, 0, {}], ["variational_log_variance", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 45, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_log_variance", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["variational_mean", 0, 0, {}], ["variational_log_variance", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}}
?
regularization_losses
trainable_variables
	variables
	keras_api
*&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
?
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
?
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "variational_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 288}}}}
?

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "variational_log_variance", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 288}}}}
?
?regularization_losses
@trainable_variables
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
trackable_list_wrapper
f
0
1
2
3
%4
&5
36
47
98
:9"
trackable_list_wrapper
f
0
1
2
3
%4
&5
36
47
98
:9"
trackable_list_wrapper
?
Clayer_metrics
regularization_losses
Dmetrics

Elayers
	variables
Fnon_trainable_variables
trainable_variables
Glayer_regularization_losses
|__call__
{_default_save_signature
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%@2conv2d/kernel
:@2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Hlayer_metrics
regularization_losses

Ilayers
trainable_variables
	variables
Jnon_trainable_variables
Kmetrics
Llayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mlayer_metrics
regularization_losses

Nlayers
trainable_variables
	variables
Onon_trainable_variables
Pmetrics
Qlayer_regularization_losses
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Rlayer_metrics
regularization_losses

Slayers
trainable_variables
	variables
Tnon_trainable_variables
Umetrics
Vlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wlayer_metrics
!regularization_losses

Xlayers
"trainable_variables
#	variables
Ynon_trainable_variables
Zmetrics
[layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@ 2conv2d_2/kernel
: 2conv2d_2/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
\layer_metrics
'regularization_losses

]layers
(trainable_variables
)	variables
^non_trainable_variables
_metrics
`layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
alayer_metrics
+regularization_losses

blayers
,trainable_variables
-	variables
cnon_trainable_variables
dmetrics
elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
flayer_metrics
/regularization_losses

glayers
0trainable_variables
1	variables
hnon_trainable_variables
imetrics
jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(	?d2variational_mean/kernel
#:!d2variational_mean/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
klayer_metrics
5regularization_losses

llayers
6trainable_variables
7	variables
mnon_trainable_variables
nmetrics
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0	?d2variational_log_variance/kernel
+:)d2variational_log_variance/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
player_metrics
;regularization_losses

qlayers
<trainable_variables
=	variables
rnon_trainable_variables
smetrics
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ulayer_metrics
?regularization_losses

vlayers
@trainable_variables
A	variables
wnon_trainable_variables
xmetrics
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
G__inference_functional_1_layer_call_and_return_conditional_losses_10530
G__inference_functional_1_layer_call_and_return_conditional_losses_10689
G__inference_functional_1_layer_call_and_return_conditional_losses_10752
G__inference_functional_1_layer_call_and_return_conditional_losses_10506?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_10238?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????--
?2?
,__inference_functional_1_layer_call_fn_10570
,__inference_functional_1_layer_call_fn_10609
,__inference_functional_1_layer_call_fn_10782
,__inference_functional_1_layer_call_fn_10767?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_10793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2d_layer_call_fn_10800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10244?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_max_pooling2d_layer_call_fn_10255?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_10818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_10261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_1_layer_call_fn_10272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_2_layer_call_fn_10836?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_2_layer_call_fn_10289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_10842?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_10847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_variational_mean_layer_call_and_return_conditional_losses_10857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_variational_mean_layer_call_fn_10864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_10874?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_variational_log_variance_layer_call_fn_10881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_lambda_layer_call_and_return_conditional_losses_10907
A__inference_lambda_layer_call_and_return_conditional_losses_10933?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lambda_layer_call_fn_10939
&__inference_lambda_layer_call_fn_10945?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
2B0
#__inference_signature_wrapper_10626input_1?
 __inference__wrapped_model_10238w
%&349:8?5
.?+
)?&
input_1?????????--
? "/?,
*
lambda ?
lambda?????????d?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10811l7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_1_layer_call_fn_10818_7?4
-?*
(?%
inputs?????????@
? " ??????????@?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10829l%&7?4
-?*
(?%
inputs?????????		@
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_2_layer_call_fn_10836_%&7?4
-?*
(?%
inputs?????????		@
? " ?????????? ?
A__inference_conv2d_layer_call_and_return_conditional_losses_10793l7?4
-?*
(?%
inputs?????????--
? "-?*
#? 
0?????????))@
? ?
&__inference_conv2d_layer_call_fn_10800_7?4
-?*
(?%
inputs?????????--
? " ??????????))@?
B__inference_flatten_layer_call_and_return_conditional_losses_10842a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_10847T7?4
-?*
(?%
inputs????????? 
? "????????????
G__inference_functional_1_layer_call_and_return_conditional_losses_10506u
%&349:@?=
6?3
)?&
input_1?????????--
p

 
? "%?"
?
0?????????d
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_10530u
%&349:@?=
6?3
)?&
input_1?????????--
p 

 
? "%?"
?
0?????????d
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_10689t
%&349:??<
5?2
(?%
inputs?????????--
p

 
? "%?"
?
0?????????d
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_10752t
%&349:??<
5?2
(?%
inputs?????????--
p 

 
? "%?"
?
0?????????d
? ?
,__inference_functional_1_layer_call_fn_10570h
%&349:@?=
6?3
)?&
input_1?????????--
p

 
? "??????????d?
,__inference_functional_1_layer_call_fn_10609h
%&349:@?=
6?3
)?&
input_1?????????--
p 

 
? "??????????d?
,__inference_functional_1_layer_call_fn_10767g
%&349:??<
5?2
(?%
inputs?????????--
p

 
? "??????????d?
,__inference_functional_1_layer_call_fn_10782g
%&349:??<
5?2
(?%
inputs?????????--
p 

 
? "??????????d?
A__inference_lambda_layer_call_and_return_conditional_losses_10907?b?_
X?U
K?H
"?
inputs/0?????????d
"?
inputs/1?????????d

 
p
? "%?"
?
0?????????d
? ?
A__inference_lambda_layer_call_and_return_conditional_losses_10933?b?_
X?U
K?H
"?
inputs/0?????????d
"?
inputs/1?????????d

 
p 
? "%?"
?
0?????????d
? ?
&__inference_lambda_layer_call_fn_10939~b?_
X?U
K?H
"?
inputs/0?????????d
"?
inputs/1?????????d

 
p
? "??????????d?
&__inference_lambda_layer_call_fn_10945~b?_
X?U
K?H
"?
inputs/0?????????d
"?
inputs/1?????????d

 
p 
? "??????????d?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_10261?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_10272?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10278?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_10289?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10244?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_10255?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_10626?
%&349:C?@
? 
9?6
4
input_1)?&
input_1?????????--"/?,
*
lambda ?
lambda?????????d?
S__inference_variational_log_variance_layer_call_and_return_conditional_losses_10874]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
8__inference_variational_log_variance_layer_call_fn_10881P9:0?-
&?#
!?
inputs??????????
? "??????????d?
K__inference_variational_mean_layer_call_and_return_conditional_losses_10857]340?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
0__inference_variational_mean_layer_call_fn_10864P340?-
&?#
!?
inputs??????????
? "??????????d