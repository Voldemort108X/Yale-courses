═я
ЛБ
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02v2.3.0-rc2-23-gb36436b0878▄к

њ
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@d*(
shared_nameconv2d_transpose/kernel
І
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:@d*
dtype0
ѓ
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:@*
dtype0
ќ
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_1/kernel
Ј
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
: @*
dtype0
є
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
: *
dtype0
ќ
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_2/kernel
Ј
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: *
dtype0
є
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0
ќ
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_3/kernel
Ј
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:*
dtype0
є
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
џ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н
value╦B╚ B┴
┐
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
 
8
0
1
2
3
'4
(5
16
27
8
0
1
2
3
'4
(5
16
27
Г
7layer_metrics

regularization_losses
8metrics

9layers
	variables
:non_trainable_variables
trainable_variables
;layer_regularization_losses
 
 
 
 
Г
<layer_metrics
regularization_losses

=layers
trainable_variables
	variables
>non_trainable_variables
?metrics
@layer_regularization_losses
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
Alayer_metrics
regularization_losses

Blayers
trainable_variables
	variables
Cnon_trainable_variables
Dmetrics
Elayer_regularization_losses
 
 
 
Г
Flayer_metrics
regularization_losses

Glayers
trainable_variables
	variables
Hnon_trainable_variables
Imetrics
Jlayer_regularization_losses
ec
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
Klayer_metrics
regularization_losses

Llayers
 trainable_variables
!	variables
Mnon_trainable_variables
Nmetrics
Olayer_regularization_losses
 
 
 
Г
Player_metrics
#regularization_losses

Qlayers
$trainable_variables
%	variables
Rnon_trainable_variables
Smetrics
Tlayer_regularization_losses
ec
VARIABLE_VALUEconv2d_transpose_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
Г
Ulayer_metrics
)regularization_losses

Vlayers
*trainable_variables
+	variables
Wnon_trainable_variables
Xmetrics
Ylayer_regularization_losses
 
 
 
Г
Zlayer_metrics
-regularization_losses

[layers
.trainable_variables
/	variables
\non_trainable_variables
]metrics
^layer_regularization_losses
ec
VARIABLE_VALUEconv2d_transpose_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
Г
_layer_metrics
3regularization_losses

`layers
4trainable_variables
5	variables
anon_trainable_variables
bmetrics
clayer_regularization_losses
 
 
?
0
1
2
3
4
5
6
7
	8
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
z
serving_default_input_2Placeholder*'
_output_shapes
:         d*
dtype0*
shape:         d
Ќ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference_signature_wrapper_5397
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
љ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOpConst*
Tin
2
*
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
GPU2*0J 8ѓ *&
f!R
__inference__traced_save_5725
в
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/bias*
Tin
2	*
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
GPU2*0J 8ѓ *)
f$R"
 __inference__traced_restore_5759Щє

Ї*
Ќ
F__inference_functional_3_layer_call_and_return_conditional_losses_5338

inputs,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCall┌
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_52282
reshape/PartitionedCallЉ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48572*
(conv2d_transpose/StatefulPartitionedCallЕ
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_48902
up_sampling2d/PartitionedCallЦ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49732,
*conv2d_transpose_1/StatefulPartitionedCall▒
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_50062!
up_sampling2d_1/PartitionedCallД
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_50892,
*conv2d_transpose_2/StatefulPartitionedCall▒
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_51222!
up_sampling2d_2/PartitionedCallД
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_52052,
*conv2d_transpose_3/StatefulPartitionedCallМ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
К

Л
+__inference_functional_3_layer_call_fn_5382
input_2
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identityѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_53712
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         d
!
_user_specified_name	input_2
Ъ
B
&__inference_reshape_layer_call_fn_5678

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_52282
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
є
e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_5122

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
─

л
+__inference_functional_3_layer_call_fn_5646

inputs
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_53382
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Р
]
A__inference_reshape_layer_call_and_return_conditional_losses_5673

inputs
identityD
ShapeShapeinputs*
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         d2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ў
д
1__inference_conv2d_transpose_2_layer_call_fn_5094

inputs
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_kernelconv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_50892
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
щ%
п
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4816

inputs;
7conv2d_transpose_readvariableop_conv2d_transpose_kernel0
,biasadd_readvariableop_conv2d_transpose_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┬
conv2d_transpose/ReadVariableOpReadVariableOp7conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
conv2d_transposeЎ
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           d:::i e
A
_output_shapes/
-:+                           d
 
_user_specified_nameinputs
ї
а
/__inference_conv2d_transpose_layer_call_fn_4862

inputs
conv2d_transpose_kernel
conv2d_transpose_bias
identityѕбStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48572
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           d::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           d
 
_user_specified_nameinputs
г
J
.__inference_up_sampling2d_1_layer_call_fn_5009

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_50062
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ў
д
1__inference_conv2d_transpose_3_layer_call_fn_5210

inputs
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_52052
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Л 
«
__inference__traced_save_5725
file_prefix6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6eeb6764694f470698c8a6357e8a7f8b/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┘
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*в
valueрBя	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesџ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesТ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*w
_input_shapesf
d: :@d:@: @: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@d: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::	

_output_shapes
: 
Р
]
A__inference_reshape_layer_call_and_return_conditional_losses_5228

inputs
identityD
ShapeShapeinputs*
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         d2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ў
д
1__inference_conv2d_transpose_1_layer_call_fn_4978

inputs
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_1_kernelconv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49732
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ѓ&
я
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_5164

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel2
.biasadd_readvariableop_conv2d_transpose_3_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3─
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeЏ
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
зд
┴
F__inference_functional_3_layer_call_and_return_conditional_losses_5633

inputsL
Hconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelA
=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_biasP
Lconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelE
Aconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_biasP
Lconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelE
Aconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_biasP
Lconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelE
Aconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityѕT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЈ
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         d2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shapeќ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackџ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1џ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2╚
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3Э
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackџ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackъ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1ъ
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2м
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1ш
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpх
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose╠
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOpо
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_transpose/BiasAddЊ
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_transpose/Relu}
up_sampling2d/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shapeљ
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stackћ
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1ћ
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2б
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Constќ
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulЂ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#conv2d_transpose/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:         @*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborЪ
conv2d_transpose_1/ShapeShape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shapeџ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackъ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1ъ
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2н
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3ё
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackъ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackб
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1б
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2я
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1§
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpЯ
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transposeн
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpя
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_transpose_1/BiasAddЎ
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_transpose_1/ReluЃ
up_sampling2d_1/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shapeћ
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stackў
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1ў
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2«
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Constъ
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulЅ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_1/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborА
conv2d_transpose_2/ShapeShape=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shapeџ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackъ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1ъ
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2н
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3ё
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackъ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackб
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1б
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2я
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1§
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpР
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transposeн
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpя
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/BiasAddЎ
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/ReluЃ
up_sampling2d_2/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shapeћ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackў
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1ў
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2«
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Constъ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulЅ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_2/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:         ((*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighborА
conv2d_transpose_3/ShapeShape=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shapeџ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackъ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1ъ
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2н
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :-2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3ё
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackъ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stackб
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1б
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2я
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1§
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpР
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:         --*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transposeн
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpя
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --2
conv2d_transpose_3/BiasAddЎ
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         --2
conv2d_transpose_3/ReluЂ
IdentityIdentity%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         --2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d:::::::::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
щ%
п
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4857

inputs;
7conv2d_transpose_readvariableop_conv2d_transpose_kernel0
,biasadd_readvariableop_conv2d_transpose_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┬
conv2d_transpose/ReadVariableOpReadVariableOp7conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
conv2d_transposeЎ
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           d:::i e
A
_output_shapes/
-:+                           d
 
_user_specified_nameinputs
Ѓ&
я
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_5048

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel2
.biasadd_readvariableop_conv2d_transpose_2_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3─
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeЏ
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
г
J
.__inference_up_sampling2d_2_layer_call_fn_5125

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_51222
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ѓ&
я
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4973

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel2
.biasadd_readvariableop_conv2d_transpose_1_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3─
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
conv2d_transposeЏ
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @:::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ї*
Ќ
F__inference_functional_3_layer_call_and_return_conditional_losses_5371

inputs,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCall┌
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_52282
reshape/PartitionedCallЉ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48572*
(conv2d_transpose/StatefulPartitionedCallЕ
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_48902
up_sampling2d/PartitionedCallЦ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49732,
*conv2d_transpose_1/StatefulPartitionedCall▒
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_50062!
up_sampling2d_1/PartitionedCallД
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_50892,
*conv2d_transpose_2/StatefulPartitionedCall▒
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_51222!
up_sampling2d_2/PartitionedCallД
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_52052,
*conv2d_transpose_3/StatefulPartitionedCallМ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
х╔
Ѓ
__inference__wrapped_model_4777
input_2Y
Ufunctional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelN
Jfunctional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias]
Yfunctional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelR
Nfunctional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias]
Yfunctional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelR
Nfunctional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias]
Yfunctional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelR
Nfunctional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityѕo
functional_3/reshape/ShapeShapeinput_2*
T0*
_output_shapes
:2
functional_3/reshape/Shapeъ
(functional_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_3/reshape/strided_slice/stackб
*functional_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_3/reshape/strided_slice/stack_1б
*functional_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_3/reshape/strided_slice/stack_2Я
"functional_3/reshape/strided_sliceStridedSlice#functional_3/reshape/Shape:output:01functional_3/reshape/strided_slice/stack:output:03functional_3/reshape/strided_slice/stack_1:output:03functional_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"functional_3/reshape/strided_sliceј
$functional_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/reshape/Reshape/shape/1ј
$functional_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/reshape/Reshape/shape/2ј
$functional_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2&
$functional_3/reshape/Reshape/shape/3И
"functional_3/reshape/Reshape/shapePack+functional_3/reshape/strided_slice:output:0-functional_3/reshape/Reshape/shape/1:output:0-functional_3/reshape/Reshape/shape/2:output:0-functional_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"functional_3/reshape/Reshape/shapeи
functional_3/reshape/ReshapeReshapeinput_2+functional_3/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         d2
functional_3/reshape/ReshapeЪ
#functional_3/conv2d_transpose/ShapeShape%functional_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#functional_3/conv2d_transpose/Shape░
1functional_3/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1functional_3/conv2d_transpose/strided_slice/stack┤
3functional_3/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_3/conv2d_transpose/strided_slice/stack_1┤
3functional_3/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_3/conv2d_transpose/strided_slice/stack_2ќ
+functional_3/conv2d_transpose/strided_sliceStridedSlice,functional_3/conv2d_transpose/Shape:output:0:functional_3/conv2d_transpose/strided_slice/stack:output:0<functional_3/conv2d_transpose/strided_slice/stack_1:output:0<functional_3/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+functional_3/conv2d_transpose/strided_sliceљ
%functional_3/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_3/conv2d_transpose/stack/1љ
%functional_3/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_3/conv2d_transpose/stack/2љ
%functional_3/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%functional_3/conv2d_transpose/stack/3к
#functional_3/conv2d_transpose/stackPack4functional_3/conv2d_transpose/strided_slice:output:0.functional_3/conv2d_transpose/stack/1:output:0.functional_3/conv2d_transpose/stack/2:output:0.functional_3/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#functional_3/conv2d_transpose/stack┤
3functional_3/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose/strided_slice_1/stackИ
5functional_3/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose/strided_slice_1/stack_1И
5functional_3/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose/strided_slice_1/stack_2а
-functional_3/conv2d_transpose/strided_slice_1StridedSlice,functional_3/conv2d_transpose/stack:output:0<functional_3/conv2d_transpose/strided_slice_1/stack:output:0>functional_3/conv2d_transpose/strided_slice_1/stack_1:output:0>functional_3/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose/strided_slice_1ю
=functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpUfunctional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02?
=functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpШ
.functional_3/conv2d_transpose/conv2d_transposeConv2DBackpropInput,functional_3/conv2d_transpose/stack:output:0Efunctional_3/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%functional_3/reshape/Reshape:output:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
20
.functional_3/conv2d_transpose/conv2d_transposeз
4functional_3/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpJfunctional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype026
4functional_3/conv2d_transpose/BiasAdd/ReadVariableOpі
%functional_3/conv2d_transpose/BiasAddBiasAdd7functional_3/conv2d_transpose/conv2d_transpose:output:0<functional_3/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2'
%functional_3/conv2d_transpose/BiasAdd║
"functional_3/conv2d_transpose/ReluRelu.functional_3/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         @2$
"functional_3/conv2d_transpose/Reluц
 functional_3/up_sampling2d/ShapeShape0functional_3/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d/Shapeф
.functional_3/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.functional_3/up_sampling2d/strided_slice/stack«
0functional_3/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d/strided_slice/stack_1«
0functional_3/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d/strided_slice/stack_2­
(functional_3/up_sampling2d/strided_sliceStridedSlice)functional_3/up_sampling2d/Shape:output:07functional_3/up_sampling2d/strided_slice/stack:output:09functional_3/up_sampling2d/strided_slice/stack_1:output:09functional_3/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(functional_3/up_sampling2d/strided_sliceЋ
 functional_3/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2"
 functional_3/up_sampling2d/Const╩
functional_3/up_sampling2d/mulMul1functional_3/up_sampling2d/strided_slice:output:0)functional_3/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2 
functional_3/up_sampling2d/mulх
7functional_3/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor0functional_3/conv2d_transpose/Relu:activations:0"functional_3/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:         @*
half_pixel_centers(29
7functional_3/up_sampling2d/resize/ResizeNearestNeighborк
%functional_3/conv2d_transpose_1/ShapeShapeHfunctional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_1/Shape┤
3functional_3/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_1/strided_slice/stackИ
5functional_3/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_1/strided_slice/stack_1И
5functional_3/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_1/strided_slice/stack_2б
-functional_3/conv2d_transpose_1/strided_sliceStridedSlice.functional_3/conv2d_transpose_1/Shape:output:0<functional_3/conv2d_transpose_1/strided_slice/stack:output:0>functional_3/conv2d_transpose_1/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_1/strided_sliceћ
'functional_3/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_1/stack/1ћ
'functional_3/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_1/stack/2ћ
'functional_3/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'functional_3/conv2d_transpose_1/stack/3м
%functional_3/conv2d_transpose_1/stackPack6functional_3/conv2d_transpose_1/strided_slice:output:00functional_3/conv2d_transpose_1/stack/1:output:00functional_3/conv2d_transpose_1/stack/2:output:00functional_3/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_1/stackИ
5functional_3/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_1/strided_slice_1/stack╝
7functional_3/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_1/strided_slice_1/stack_1╝
7functional_3/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_1/strided_slice_1/stack_2г
/functional_3/conv2d_transpose_1/strided_slice_1StridedSlice.functional_3/conv2d_transpose_1/stack:output:0>functional_3/conv2d_transpose_1/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_1/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_1/strided_slice_1ц
?functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02A
?functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOpА
0functional_3/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_1/stack:output:0Gfunctional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Hfunctional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
22
0functional_3/conv2d_transpose_1/conv2d_transposeч
6functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype028
6functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOpњ
'functional_3/conv2d_transpose_1/BiasAddBiasAdd9functional_3/conv2d_transpose_1/conv2d_transpose:output:0>functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2)
'functional_3/conv2d_transpose_1/BiasAdd└
$functional_3/conv2d_transpose_1/ReluRelu0functional_3/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:          2&
$functional_3/conv2d_transpose_1/Reluф
"functional_3/up_sampling2d_1/ShapeShape2functional_3/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2$
"functional_3/up_sampling2d_1/Shape«
0functional_3/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d_1/strided_slice/stack▓
2functional_3/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_1/strided_slice/stack_1▓
2functional_3/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_1/strided_slice/stack_2Ч
*functional_3/up_sampling2d_1/strided_sliceStridedSlice+functional_3/up_sampling2d_1/Shape:output:09functional_3/up_sampling2d_1/strided_slice/stack:output:0;functional_3/up_sampling2d_1/strided_slice/stack_1:output:0;functional_3/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_3/up_sampling2d_1/strided_sliceЎ
"functional_3/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_3/up_sampling2d_1/Constм
 functional_3/up_sampling2d_1/mulMul3functional_3/up_sampling2d_1/strided_slice:output:0+functional_3/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d_1/mulй
9functional_3/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor2functional_3/conv2d_transpose_1/Relu:activations:0$functional_3/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2;
9functional_3/up_sampling2d_1/resize/ResizeNearestNeighbor╚
%functional_3/conv2d_transpose_2/ShapeShapeJfunctional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_2/Shape┤
3functional_3/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_2/strided_slice/stackИ
5functional_3/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_2/strided_slice/stack_1И
5functional_3/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_2/strided_slice/stack_2б
-functional_3/conv2d_transpose_2/strided_sliceStridedSlice.functional_3/conv2d_transpose_2/Shape:output:0<functional_3/conv2d_transpose_2/strided_slice/stack:output:0>functional_3/conv2d_transpose_2/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_2/strided_sliceћ
'functional_3/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/1ћ
'functional_3/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/2ћ
'functional_3/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/3м
%functional_3/conv2d_transpose_2/stackPack6functional_3/conv2d_transpose_2/strided_slice:output:00functional_3/conv2d_transpose_2/stack/1:output:00functional_3/conv2d_transpose_2/stack/2:output:00functional_3/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_2/stackИ
5functional_3/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_2/strided_slice_1/stack╝
7functional_3/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_2/strided_slice_1/stack_1╝
7functional_3/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_2/strided_slice_1/stack_2г
/functional_3/conv2d_transpose_2/strided_slice_1StridedSlice.functional_3/conv2d_transpose_2/stack:output:0>functional_3/conv2d_transpose_2/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_2/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_2/strided_slice_1ц
?functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02A
?functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOpБ
0functional_3/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_2/stack:output:0Gfunctional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0Jfunctional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
22
0functional_3/conv2d_transpose_2/conv2d_transposeч
6functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype028
6functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOpњ
'functional_3/conv2d_transpose_2/BiasAddBiasAdd9functional_3/conv2d_transpose_2/conv2d_transpose:output:0>functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2)
'functional_3/conv2d_transpose_2/BiasAdd└
$functional_3/conv2d_transpose_2/ReluRelu0functional_3/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2&
$functional_3/conv2d_transpose_2/Reluф
"functional_3/up_sampling2d_2/ShapeShape2functional_3/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2$
"functional_3/up_sampling2d_2/Shape«
0functional_3/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d_2/strided_slice/stack▓
2functional_3/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_2/strided_slice/stack_1▓
2functional_3/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_2/strided_slice/stack_2Ч
*functional_3/up_sampling2d_2/strided_sliceStridedSlice+functional_3/up_sampling2d_2/Shape:output:09functional_3/up_sampling2d_2/strided_slice/stack:output:0;functional_3/up_sampling2d_2/strided_slice/stack_1:output:0;functional_3/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_3/up_sampling2d_2/strided_sliceЎ
"functional_3/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_3/up_sampling2d_2/Constм
 functional_3/up_sampling2d_2/mulMul3functional_3/up_sampling2d_2/strided_slice:output:0+functional_3/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d_2/mulй
9functional_3/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor2functional_3/conv2d_transpose_2/Relu:activations:0$functional_3/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:         ((*
half_pixel_centers(2;
9functional_3/up_sampling2d_2/resize/ResizeNearestNeighbor╚
%functional_3/conv2d_transpose_3/ShapeShapeJfunctional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_3/Shape┤
3functional_3/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_3/strided_slice/stackИ
5functional_3/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_3/strided_slice/stack_1И
5functional_3/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_3/strided_slice/stack_2б
-functional_3/conv2d_transpose_3/strided_sliceStridedSlice.functional_3/conv2d_transpose_3/Shape:output:0<functional_3/conv2d_transpose_3/strided_slice/stack:output:0>functional_3/conv2d_transpose_3/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_3/strided_sliceћ
'functional_3/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :-2)
'functional_3/conv2d_transpose_3/stack/1ћ
'functional_3/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2)
'functional_3/conv2d_transpose_3/stack/2ћ
'functional_3/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_3/stack/3м
%functional_3/conv2d_transpose_3/stackPack6functional_3/conv2d_transpose_3/strided_slice:output:00functional_3/conv2d_transpose_3/stack/1:output:00functional_3/conv2d_transpose_3/stack/2:output:00functional_3/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_3/stackИ
5functional_3/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_3/strided_slice_1/stack╝
7functional_3/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_3/strided_slice_1/stack_1╝
7functional_3/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_3/strided_slice_1/stack_2г
/functional_3/conv2d_transpose_3/strided_slice_1StridedSlice.functional_3/conv2d_transpose_3/stack:output:0>functional_3/conv2d_transpose_3/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_3/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_3/strided_slice_1ц
?functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02A
?functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOpБ
0functional_3/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_3/stack:output:0Gfunctional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0Jfunctional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:         --*
paddingVALID*
strides
22
0functional_3/conv2d_transpose_3/conv2d_transposeч
6functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype028
6functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOpњ
'functional_3/conv2d_transpose_3/BiasAddBiasAdd9functional_3/conv2d_transpose_3/conv2d_transpose:output:0>functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --2)
'functional_3/conv2d_transpose_3/BiasAdd└
$functional_3/conv2d_transpose_3/ReluRelu0functional_3/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         --2&
$functional_3/conv2d_transpose_3/Reluј
IdentityIdentity2functional_3/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         --2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d:::::::::P L
'
_output_shapes
:         d
!
_user_specified_name	input_2
─

л
+__inference_functional_3_layer_call_fn_5659

inputs
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_53712
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
зд
┴
F__inference_functional_3_layer_call_and_return_conditional_losses_5515

inputsL
Hconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelA
=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_biasP
Lconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelE
Aconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_biasP
Lconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelE
Aconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_biasP
Lconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelE
Aconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityѕT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЈ
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         d2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shapeќ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackџ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1џ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2╚
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3Э
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackџ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackъ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1ъ
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2м
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1ш
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpх
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose╠
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOpо
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_transpose/BiasAddЊ
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_transpose/Relu}
up_sampling2d/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shapeљ
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stackћ
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1ћ
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2б
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Constќ
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulЂ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#conv2d_transpose/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:         @*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborЪ
conv2d_transpose_1/ShapeShape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shapeџ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackъ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1ъ
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2н
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3ё
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackъ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackб
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1б
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2я
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1§
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpЯ
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transposeн
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpя
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_transpose_1/BiasAddЎ
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_transpose_1/ReluЃ
up_sampling2d_1/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shapeћ
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stackў
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1ў
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2«
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Constъ
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulЅ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_1/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:          *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborА
conv2d_transpose_2/ShapeShape=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shapeџ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackъ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1ъ
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2н
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3ё
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackъ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackб
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1б
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2я
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1§
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpР
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transposeн
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpя
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/BiasAddЎ
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/ReluЃ
up_sampling2d_2/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shapeћ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackў
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1ў
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2«
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Constъ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulЅ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_2/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:         ((*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighborА
conv2d_transpose_3/ShapeShape=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shapeџ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackъ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1ъ
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2н
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :-2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3ё
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackъ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stackб
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1б
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2я
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1§
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpР
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:         --*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transposeн
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpя
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --2
conv2d_transpose_3/BiasAddЎ
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         --2
conv2d_transpose_3/ReluЂ
IdentityIdentity%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         --2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d:::::::::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ч&
Ь
 __inference__traced_restore_5759
file_prefix,
(assignvariableop_conv2d_transpose_kernel,
(assignvariableop_1_conv2d_transpose_bias0
,assignvariableop_2_conv2d_transpose_1_kernel.
*assignvariableop_3_conv2d_transpose_1_bias0
,assignvariableop_4_conv2d_transpose_2_kernel.
*assignvariableop_5_conv2d_transpose_2_bias0
,assignvariableop_6_conv2d_transpose_3_kernel.
*assignvariableop_7_conv2d_transpose_3_bias

identity_9ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7▀
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*в
valueрBя	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesп
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityД
AssignVariableOpAssignVariableOp(assignvariableop_conv2d_transpose_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Г
AssignVariableOp_1AssignVariableOp(assignvariableop_1_conv2d_transpose_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2▒
AssignVariableOp_2AssignVariableOp,assignvariableop_2_conv2d_transpose_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2d_transpose_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5»
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6▒
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7»
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpј

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8ђ

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
є
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_5006

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
є
e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_5107

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
е
H
,__inference_up_sampling2d_layer_call_fn_4893

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_48902
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ѓ&
я
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4932

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel2
.biasadd_readvariableop_conv2d_transpose_1_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3─
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
conv2d_transposeЏ
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @:::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
є
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_4991

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
љ*
ў
F__inference_functional_3_layer_call_and_return_conditional_losses_5295
input_2,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCall█
reshape/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_52282
reshape/PartitionedCallЉ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48572*
(conv2d_transpose/StatefulPartitionedCallЕ
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_48902
up_sampling2d/PartitionedCallЦ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49732,
*conv2d_transpose_1/StatefulPartitionedCall▒
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_50062!
up_sampling2d_1/PartitionedCallД
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_50892,
*conv2d_transpose_2/StatefulPartitionedCall▒
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_51222!
up_sampling2d_2/PartitionedCallД
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_52052,
*conv2d_transpose_3/StatefulPartitionedCallМ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:P L
'
_output_shapes
:         d
!
_user_specified_name	input_2
Ѓ&
я
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_5205

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel2
.biasadd_readvariableop_conv2d_transpose_3_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3─
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeЏ
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
з	
╚
"__inference_signature_wrapper_5397
input_2
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *(
f#R!
__inference__wrapped_model_47772
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         --2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         d
!
_user_specified_name	input_2
Ѓ&
я
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_5089

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel2
.biasadd_readvariableop_conv2d_transpose_2_bias
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
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
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3─
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeЏ
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Reluђ
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
љ*
ў
F__inference_functional_3_layer_call_and_return_conditional_losses_5315
input_2,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCall█
reshape/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_52282
reshape/PartitionedCallЉ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48572*
(conv2d_transpose/StatefulPartitionedCallЕ
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_48902
up_sampling2d/PartitionedCallЦ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_49732,
*conv2d_transpose_1/StatefulPartitionedCall▒
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_50062!
up_sampling2d_1/PartitionedCallД
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_50892,
*conv2d_transpose_2/StatefulPartitionedCall▒
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_51222!
up_sampling2d_2/PartitionedCallД
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_52052,
*conv2d_transpose_3/StatefulPartitionedCallМ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:P L
'
_output_shapes
:         d
!
_user_specified_name	input_2
К

Л
+__inference_functional_3_layer_call_fn_5349
input_2
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identityѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_53382
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         d
!
_user_specified_name	input_2
ё
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4875

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ё
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4890

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*й
serving_defaultЕ
;
input_20
serving_default_input_2:0         dN
conv2d_transpose_38
StatefulPartitionedCall:0         --tensorflow/serving/predict:дљ
├O
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api

signatures
*d&call_and_return_all_conditional_losses
e_default_save_signature
f__call__"фL
_tf_keras_networkјL{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}, "name": "reshape", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}, "name": "reshape", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}}}
ь"Ж
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
ш
regularization_losses
trainable_variables
	variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"Т
_tf_keras_layer╠{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}}
н	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*i&call_and_return_all_conditional_losses
j__call__"»
_tf_keras_layerЋ{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 100}}}}
┼
regularization_losses
trainable_variables
	variables
	keras_api
*k&call_and_return_all_conditional_losses
l__call__"Х
_tf_keras_layerю{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
О	

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
*m&call_and_return_all_conditional_losses
n__call__"▓
_tf_keras_layerў{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
╔
#regularization_losses
$trainable_variables
%	variables
&	keras_api
*o&call_and_return_all_conditional_losses
p__call__"║
_tf_keras_layerа{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
О	

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
*q&call_and_return_all_conditional_losses
r__call__"▓
_tf_keras_layerў{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
╔
-regularization_losses
.trainable_variables
/	variables
0	keras_api
*s&call_and_return_all_conditional_losses
t__call__"║
_tf_keras_layerа{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
о	

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
*u&call_and_return_all_conditional_losses
v__call__"▒
_tf_keras_layerЌ{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
 "
trackable_list_wrapper
X
0
1
2
3
'4
(5
16
27"
trackable_list_wrapper
X
0
1
2
3
'4
(5
16
27"
trackable_list_wrapper
╩
7layer_metrics

regularization_losses
8metrics

9layers
	variables
:non_trainable_variables
trainable_variables
;layer_regularization_losses
f__call__
e_default_save_signature
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
,
wserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
<layer_metrics
regularization_losses

=layers
trainable_variables
	variables
>non_trainable_variables
?metrics
@layer_regularization_losses
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
1:/@d2conv2d_transpose/kernel
#:!@2conv2d_transpose/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
Alayer_metrics
regularization_losses

Blayers
trainable_variables
	variables
Cnon_trainable_variables
Dmetrics
Elayer_regularization_losses
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Flayer_metrics
regularization_losses

Glayers
trainable_variables
	variables
Hnon_trainable_variables
Imetrics
Jlayer_regularization_losses
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
3:1 @2conv2d_transpose_1/kernel
%:# 2conv2d_transpose_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
Klayer_metrics
regularization_losses

Llayers
 trainable_variables
!	variables
Mnon_trainable_variables
Nmetrics
Olayer_regularization_losses
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Player_metrics
#regularization_losses

Qlayers
$trainable_variables
%	variables
Rnon_trainable_variables
Smetrics
Tlayer_regularization_losses
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
Г
Ulayer_metrics
)regularization_losses

Vlayers
*trainable_variables
+	variables
Wnon_trainable_variables
Xmetrics
Ylayer_regularization_losses
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Zlayer_metrics
-regularization_losses

[layers
.trainable_variables
/	variables
\non_trainable_variables
]metrics
^layer_regularization_losses
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
3:12conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
Г
_layer_metrics
3regularization_losses

`layers
4trainable_variables
5	variables
anon_trainable_variables
bmetrics
clayer_regularization_losses
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
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
Т2с
F__inference_functional_3_layer_call_and_return_conditional_losses_5515
F__inference_functional_3_layer_call_and_return_conditional_losses_5295
F__inference_functional_3_layer_call_and_return_conditional_losses_5633
F__inference_functional_3_layer_call_and_return_conditional_losses_5315└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
П2┌
__inference__wrapped_model_4777Х
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_2         d
Щ2э
+__inference_functional_3_layer_call_fn_5659
+__inference_functional_3_layer_call_fn_5349
+__inference_functional_3_layer_call_fn_5382
+__inference_functional_3_layer_call_fn_5646└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
A__inference_reshape_layer_call_and_return_conditional_losses_5673б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_reshape_layer_call_fn_5678б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Е2д
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4816О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           d
ј2І
/__inference_conv2d_transpose_layer_call_fn_4862О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           d
»2г
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4875Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ћ2Љ
,__inference_up_sampling2d_layer_call_fn_4893Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ф2е
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4932О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
љ2Ї
1__inference_conv2d_transpose_1_layer_call_fn_4978О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
▒2«
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_4991Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ќ2Њ
.__inference_up_sampling2d_1_layer_call_fn_5009Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ф2е
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_5048О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
љ2Ї
1__inference_conv2d_transpose_2_layer_call_fn_5094О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
▒2«
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_5107Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ќ2Њ
.__inference_up_sampling2d_2_layer_call_fn_5125Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ф2е
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_5164О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
љ2Ї
1__inference_conv2d_transpose_3_layer_call_fn_5210О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
1B/
"__inference_signature_wrapper_5397input_2▒
__inference__wrapped_model_4777Ї'(120б-
&б#
!і
input_2         d
ф "OфL
J
conv2d_transpose_34і1
conv2d_transpose_3         --р
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4932љIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                            
џ ╣
1__inference_conv2d_transpose_1_layer_call_fn_4978ЃIбF
?б<
:і7
inputs+                           @
ф "2і/+                            р
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_5048љ'(IбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                           
џ ╣
1__inference_conv2d_transpose_2_layer_call_fn_5094Ѓ'(IбF
?б<
:і7
inputs+                            
ф "2і/+                           р
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_5164љ12IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╣
1__inference_conv2d_transpose_3_layer_call_fn_5210Ѓ12IбF
?б<
:і7
inputs+                           
ф "2і/+                           ▀
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4816љIбF
?б<
:і7
inputs+                           d
ф "?б<
5і2
0+                           @
џ и
/__inference_conv2d_transpose_layer_call_fn_4862ЃIбF
?б<
:і7
inputs+                           d
ф "2і/+                           @л
F__inference_functional_3_layer_call_and_return_conditional_losses_5295Ё'(128б5
.б+
!і
input_2         d
p

 
ф "?б<
5і2
0+                           
џ л
F__inference_functional_3_layer_call_and_return_conditional_losses_5315Ё'(128б5
.б+
!і
input_2         d
p 

 
ф "?б<
5і2
0+                           
џ ╝
F__inference_functional_3_layer_call_and_return_conditional_losses_5515r'(127б4
-б*
 і
inputs         d
p

 
ф "-б*
#і 
0         --
џ ╝
F__inference_functional_3_layer_call_and_return_conditional_losses_5633r'(127б4
-б*
 і
inputs         d
p 

 
ф "-б*
#і 
0         --
џ Д
+__inference_functional_3_layer_call_fn_5349x'(128б5
.б+
!і
input_2         d
p

 
ф "2і/+                           Д
+__inference_functional_3_layer_call_fn_5382x'(128б5
.б+
!і
input_2         d
p 

 
ф "2і/+                           д
+__inference_functional_3_layer_call_fn_5646w'(127б4
-б*
 і
inputs         d
p

 
ф "2і/+                           д
+__inference_functional_3_layer_call_fn_5659w'(127б4
-б*
 і
inputs         d
p 

 
ф "2і/+                           Ц
A__inference_reshape_layer_call_and_return_conditional_losses_5673`/б,
%б"
 і
inputs         d
ф "-б*
#і 
0         d
џ }
&__inference_reshape_layer_call_fn_5678S/б,
%б"
 і
inputs         d
ф " і         d┐
"__inference_signature_wrapper_5397ў'(12;б8
б 
1ф.
,
input_2!і
input_2         d"OфL
J
conv2d_transpose_34і1
conv2d_transpose_3         --В
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_4991ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_up_sampling2d_1_layer_call_fn_5009ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    В
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_5107ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_up_sampling2d_2_layer_call_fn_5125ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ж
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4875ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┬
,__inference_up_sampling2d_layer_call_fn_4893ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    