ØÑ!
Ñ£
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
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ó
|
training_6/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_6/Adam/iter
u
(training_6/Adam/iter/Read/ReadVariableOpReadVariableOptraining_6/Adam/iter*
_output_shapes
: *
dtype0	

training_6/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_6/Adam/beta_1
y
*training_6/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_6/Adam/beta_1*
_output_shapes
: *
dtype0

training_6/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_6/Adam/beta_2
y
*training_6/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_6/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_6/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_6/Adam/decay
w
)training_6/Adam/decay/Read/ReadVariableOpReadVariableOptraining_6/Adam/decay*
_output_shapes
: *
dtype0

training_6/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_6/Adam/learning_rate

1training_6/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_6/Adam/learning_rate*
_output_shapes
: *
dtype0
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

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

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

variational_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 d*(
shared_namevariational_mean/kernel

+variational_mean/kernel/Read/ReadVariableOpReadVariableOpvariational_mean/kernel*
_output_shapes
:	 d*
dtype0

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

variational_log_variance/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 d*0
shared_name!variational_log_variance/kernel

3variational_log_variance/kernel/Read/ReadVariableOpReadVariableOpvariational_log_variance/kernel*
_output_shapes
:	 d*
dtype0

variational_log_variance/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namevariational_log_variance/bias

1variational_log_variance/bias/Read/ReadVariableOpReadVariableOpvariational_log_variance/bias*
_output_shapes
:d*
dtype0

conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@d*(
shared_nameconv2d_transpose/kernel

+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:@d*
dtype0

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

conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_1/kernel

-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
: @*
dtype0

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

conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_2/kernel

-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: *
dtype0

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

conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_3/kernel

-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:*
dtype0

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
¢
training_6/Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_6/Adam/conv2d/kernel/m

3training_6/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0

training_6/Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining_6/Adam/conv2d/bias/m

1training_6/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d/bias/m*
_output_shapes
:@*
dtype0
¦
!training_6/Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!training_6/Adam/conv2d_1/kernel/m

5training_6/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!training_6/Adam/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0

training_6/Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_6/Adam/conv2d_1/bias/m

3training_6/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
¦
!training_6/Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!training_6/Adam/conv2d_2/kernel/m

5training_6/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp!training_6/Adam/conv2d_2/kernel/m*&
_output_shapes
:@ *
dtype0

training_6/Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_6/Adam/conv2d_2/bias/m

3training_6/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d_2/bias/m*
_output_shapes
: *
dtype0
¯
)training_6/Adam/variational_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 d*:
shared_name+)training_6/Adam/variational_mean/kernel/m
¨
=training_6/Adam/variational_mean/kernel/m/Read/ReadVariableOpReadVariableOp)training_6/Adam/variational_mean/kernel/m*
_output_shapes
:	 d*
dtype0
¦
'training_6/Adam/variational_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*8
shared_name)'training_6/Adam/variational_mean/bias/m

;training_6/Adam/variational_mean/bias/m/Read/ReadVariableOpReadVariableOp'training_6/Adam/variational_mean/bias/m*
_output_shapes
:d*
dtype0
¿
1training_6/Adam/variational_log_variance/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 d*B
shared_name31training_6/Adam/variational_log_variance/kernel/m
¸
Etraining_6/Adam/variational_log_variance/kernel/m/Read/ReadVariableOpReadVariableOp1training_6/Adam/variational_log_variance/kernel/m*
_output_shapes
:	 d*
dtype0
¶
/training_6/Adam/variational_log_variance/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*@
shared_name1/training_6/Adam/variational_log_variance/bias/m
¯
Ctraining_6/Adam/variational_log_variance/bias/m/Read/ReadVariableOpReadVariableOp/training_6/Adam/variational_log_variance/bias/m*
_output_shapes
:d*
dtype0
¶
)training_6/Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@d*:
shared_name+)training_6/Adam/conv2d_transpose/kernel/m
¯
=training_6/Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose/kernel/m*&
_output_shapes
:@d*
dtype0
¦
'training_6/Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'training_6/Adam/conv2d_transpose/bias/m

;training_6/Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp'training_6/Adam/conv2d_transpose/bias/m*
_output_shapes
:@*
dtype0
º
+training_6/Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*<
shared_name-+training_6/Adam/conv2d_transpose_1/kernel/m
³
?training_6/Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp+training_6/Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
: @*
dtype0
ª
)training_6/Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)training_6/Adam/conv2d_transpose_1/bias/m
£
=training_6/Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose_1/bias/m*
_output_shapes
: *
dtype0
º
+training_6/Adam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+training_6/Adam/conv2d_transpose_2/kernel/m
³
?training_6/Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp+training_6/Adam/conv2d_transpose_2/kernel/m*&
_output_shapes
: *
dtype0
ª
)training_6/Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)training_6/Adam/conv2d_transpose_2/bias/m
£
=training_6/Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose_2/bias/m*
_output_shapes
:*
dtype0
º
+training_6/Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+training_6/Adam/conv2d_transpose_3/kernel/m
³
?training_6/Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp+training_6/Adam/conv2d_transpose_3/kernel/m*&
_output_shapes
:*
dtype0
ª
)training_6/Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)training_6/Adam/conv2d_transpose_3/bias/m
£
=training_6/Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose_3/bias/m*
_output_shapes
:*
dtype0
¢
training_6/Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_6/Adam/conv2d/kernel/v

3training_6/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0

training_6/Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining_6/Adam/conv2d/bias/v

1training_6/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d/bias/v*
_output_shapes
:@*
dtype0
¦
!training_6/Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!training_6/Adam/conv2d_1/kernel/v

5training_6/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!training_6/Adam/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0

training_6/Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_6/Adam/conv2d_1/bias/v

3training_6/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
¦
!training_6/Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!training_6/Adam/conv2d_2/kernel/v

5training_6/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp!training_6/Adam/conv2d_2/kernel/v*&
_output_shapes
:@ *
dtype0

training_6/Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_6/Adam/conv2d_2/bias/v

3training_6/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining_6/Adam/conv2d_2/bias/v*
_output_shapes
: *
dtype0
¯
)training_6/Adam/variational_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 d*:
shared_name+)training_6/Adam/variational_mean/kernel/v
¨
=training_6/Adam/variational_mean/kernel/v/Read/ReadVariableOpReadVariableOp)training_6/Adam/variational_mean/kernel/v*
_output_shapes
:	 d*
dtype0
¦
'training_6/Adam/variational_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*8
shared_name)'training_6/Adam/variational_mean/bias/v

;training_6/Adam/variational_mean/bias/v/Read/ReadVariableOpReadVariableOp'training_6/Adam/variational_mean/bias/v*
_output_shapes
:d*
dtype0
¿
1training_6/Adam/variational_log_variance/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 d*B
shared_name31training_6/Adam/variational_log_variance/kernel/v
¸
Etraining_6/Adam/variational_log_variance/kernel/v/Read/ReadVariableOpReadVariableOp1training_6/Adam/variational_log_variance/kernel/v*
_output_shapes
:	 d*
dtype0
¶
/training_6/Adam/variational_log_variance/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*@
shared_name1/training_6/Adam/variational_log_variance/bias/v
¯
Ctraining_6/Adam/variational_log_variance/bias/v/Read/ReadVariableOpReadVariableOp/training_6/Adam/variational_log_variance/bias/v*
_output_shapes
:d*
dtype0
¶
)training_6/Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@d*:
shared_name+)training_6/Adam/conv2d_transpose/kernel/v
¯
=training_6/Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose/kernel/v*&
_output_shapes
:@d*
dtype0
¦
'training_6/Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'training_6/Adam/conv2d_transpose/bias/v

;training_6/Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp'training_6/Adam/conv2d_transpose/bias/v*
_output_shapes
:@*
dtype0
º
+training_6/Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*<
shared_name-+training_6/Adam/conv2d_transpose_1/kernel/v
³
?training_6/Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp+training_6/Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
: @*
dtype0
ª
)training_6/Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)training_6/Adam/conv2d_transpose_1/bias/v
£
=training_6/Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose_1/bias/v*
_output_shapes
: *
dtype0
º
+training_6/Adam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+training_6/Adam/conv2d_transpose_2/kernel/v
³
?training_6/Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp+training_6/Adam/conv2d_transpose_2/kernel/v*&
_output_shapes
: *
dtype0
ª
)training_6/Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)training_6/Adam/conv2d_transpose_2/bias/v
£
=training_6/Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose_2/bias/v*
_output_shapes
:*
dtype0
º
+training_6/Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+training_6/Adam/conv2d_transpose_3/kernel/v
³
?training_6/Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp+training_6/Adam/conv2d_transpose_3/kernel/v*&
_output_shapes
:*
dtype0
ª
)training_6/Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)training_6/Adam/conv2d_transpose_3/bias/v
£
=training_6/Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOp)training_6/Adam/conv2d_transpose_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ìs
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*§s
valuesBs Bs
Ì
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
 
ä
layer-0

layer_with_weights-0

layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
layer_with_weights-4
layer-9
layer-10
regularization_losses
	variables
trainable_variables
	keras_api
¯
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
 layer_with_weights-3
 layer-8
!regularization_losses
"	variables
#trainable_variables
$	keras_api
¨
%iter

&beta_1

'beta_2
	(decay
)learning_rate*mí+mî,mï-mð.mñ/mò0mó1mô2mõ3mö4m÷5mø6mù7mú8mû9mü:mý;mþ*vÿ+v,v-v.v/v0v1v2v3v4v5v6v7v8v9v:v;v
 

*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17

*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
­
<layer_metrics
regularization_losses
=metrics

>layers
	variables
?non_trainable_variables
trainable_variables
@layer_regularization_losses
 
h

*kernel
+bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

,kernel
-bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

.kernel
/bias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
R
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

0kernel
1bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

2kernel
3bias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
R
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
 
F
*0
+1
,2
-3
.4
/5
06
17
28
39
F
*0
+1
,2
-3
.4
/5
06
17
28
39
­
ilayer_metrics
regularization_losses
jmetrics

klayers
	variables
lnon_trainable_variables
trainable_variables
mlayer_regularization_losses
 
R
nregularization_losses
otrainable_variables
p	variables
q	keras_api
h

4kernel
5bias
rregularization_losses
strainable_variables
t	variables
u	keras_api
R
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
h

6kernel
7bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
T
~regularization_losses
trainable_variables
	variables
	keras_api
l

8kernel
9bias
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
l

:kernel
;bias
regularization_losses
trainable_variables
	variables
	keras_api
 
8
40
51
62
73
84
95
:6
;7
8
40
51
62
73
84
95
:6
;7
²
layer_metrics
!regularization_losses
metrics
layers
"	variables
non_trainable_variables
#trainable_variables
 layer_regularization_losses
SQ
VARIABLE_VALUEtraining_6/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_6/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_6/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_6/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_6/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEvariational_mean/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEvariational_mean/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEvariational_log_variance/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvariational_log_variance/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconv2d_transpose/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_3/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
 
 
 

*0
+1

*0
+1
²
layer_metrics
Aregularization_losses
layers
Btrainable_variables
C	variables
non_trainable_variables
metrics
 layer_regularization_losses
 
 
 
²
layer_metrics
Eregularization_losses
layers
Ftrainable_variables
G	variables
non_trainable_variables
metrics
 layer_regularization_losses
 

,0
-1

,0
-1
²
layer_metrics
Iregularization_losses
layers
Jtrainable_variables
K	variables
non_trainable_variables
 metrics
 ¡layer_regularization_losses
 
 
 
²
¢layer_metrics
Mregularization_losses
£layers
Ntrainable_variables
O	variables
¤non_trainable_variables
¥metrics
 ¦layer_regularization_losses
 

.0
/1

.0
/1
²
§layer_metrics
Qregularization_losses
¨layers
Rtrainable_variables
S	variables
©non_trainable_variables
ªmetrics
 «layer_regularization_losses
 
 
 
²
¬layer_metrics
Uregularization_losses
­layers
Vtrainable_variables
W	variables
®non_trainable_variables
¯metrics
 °layer_regularization_losses
 
 
 
²
±layer_metrics
Yregularization_losses
²layers
Ztrainable_variables
[	variables
³non_trainable_variables
´metrics
 µlayer_regularization_losses
 

00
11

00
11
²
¶layer_metrics
]regularization_losses
·layers
^trainable_variables
_	variables
¸non_trainable_variables
¹metrics
 ºlayer_regularization_losses
 

20
31

20
31
²
»layer_metrics
aregularization_losses
¼layers
btrainable_variables
c	variables
½non_trainable_variables
¾metrics
 ¿layer_regularization_losses
 
 
 
²
Àlayer_metrics
eregularization_losses
Álayers
ftrainable_variables
g	variables
Ânon_trainable_variables
Ãmetrics
 Älayer_regularization_losses
 
 
N
0

1
2
3
4
5
6
7
8
9
10
 
 
 
 
 
²
Ålayer_metrics
nregularization_losses
Ælayers
otrainable_variables
p	variables
Çnon_trainable_variables
Èmetrics
 Élayer_regularization_losses
 

40
51

40
51
²
Êlayer_metrics
rregularization_losses
Ëlayers
strainable_variables
t	variables
Ìnon_trainable_variables
Ímetrics
 Îlayer_regularization_losses
 
 
 
²
Ïlayer_metrics
vregularization_losses
Ðlayers
wtrainable_variables
x	variables
Ñnon_trainable_variables
Òmetrics
 Ólayer_regularization_losses
 

60
71

60
71
²
Ôlayer_metrics
zregularization_losses
Õlayers
{trainable_variables
|	variables
Önon_trainable_variables
×metrics
 Ølayer_regularization_losses
 
 
 
³
Ùlayer_metrics
~regularization_losses
Úlayers
trainable_variables
	variables
Ûnon_trainable_variables
Ümetrics
 Ýlayer_regularization_losses
 

80
91

80
91
µ
Þlayer_metrics
regularization_losses
ßlayers
trainable_variables
	variables
ànon_trainable_variables
ámetrics
 âlayer_regularization_losses
 
 
 
µ
ãlayer_metrics
regularization_losses
älayers
trainable_variables
	variables
ånon_trainable_variables
æmetrics
 çlayer_regularization_losses
 

:0
;1

:0
;1
µ
èlayer_metrics
regularization_losses
élayers
trainable_variables
	variables
ênon_trainable_variables
ëmetrics
 ìlayer_regularization_losses
 
 
?
0
1
2
3
4
5
6
7
 8
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
wu
VARIABLE_VALUEtraining_6/Adam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEtraining_6/Adam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!training_6/Adam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining_6/Adam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!training_6/Adam/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining_6/Adam/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/variational_mean/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'training_6/Adam/variational_mean/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1training_6/Adam/variational_log_variance/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/training_6/Adam/variational_log_variance/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE'training_6/Adam/conv2d_transpose/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_6/Adam/conv2d_transpose_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_6/Adam/conv2d_transpose_2/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose_2/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_6/Adam/conv2d_transpose_3/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose_3/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining_6/Adam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEtraining_6/Adam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!training_6/Adam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining_6/Adam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!training_6/Adam/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining_6/Adam/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/variational_mean/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'training_6/Adam/variational_mean/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1training_6/Adam/variational_log_variance/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/training_6/Adam/variational_log_variance/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE'training_6/Adam/conv2d_transpose/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_6/Adam/conv2d_transpose_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_6/Adam/conv2d_transpose_2/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose_2/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_6/Adam/conv2d_transpose_3/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training_6/Adam/conv2d_transpose_3/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ--
÷
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasvariational_mean/kernelvariational_mean/biasvariational_log_variance/kernelvariational_log_variance/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_8772
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¿
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(training_6/Adam/iter/Read/ReadVariableOp*training_6/Adam/beta_1/Read/ReadVariableOp*training_6/Adam/beta_2/Read/ReadVariableOp)training_6/Adam/decay/Read/ReadVariableOp1training_6/Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp+variational_mean/kernel/Read/ReadVariableOp)variational_mean/bias/Read/ReadVariableOp3variational_log_variance/kernel/Read/ReadVariableOp1variational_log_variance/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp3training_6/Adam/conv2d/kernel/m/Read/ReadVariableOp1training_6/Adam/conv2d/bias/m/Read/ReadVariableOp5training_6/Adam/conv2d_1/kernel/m/Read/ReadVariableOp3training_6/Adam/conv2d_1/bias/m/Read/ReadVariableOp5training_6/Adam/conv2d_2/kernel/m/Read/ReadVariableOp3training_6/Adam/conv2d_2/bias/m/Read/ReadVariableOp=training_6/Adam/variational_mean/kernel/m/Read/ReadVariableOp;training_6/Adam/variational_mean/bias/m/Read/ReadVariableOpEtraining_6/Adam/variational_log_variance/kernel/m/Read/ReadVariableOpCtraining_6/Adam/variational_log_variance/bias/m/Read/ReadVariableOp=training_6/Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp;training_6/Adam/conv2d_transpose/bias/m/Read/ReadVariableOp?training_6/Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp=training_6/Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp?training_6/Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp=training_6/Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOp?training_6/Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp=training_6/Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp3training_6/Adam/conv2d/kernel/v/Read/ReadVariableOp1training_6/Adam/conv2d/bias/v/Read/ReadVariableOp5training_6/Adam/conv2d_1/kernel/v/Read/ReadVariableOp3training_6/Adam/conv2d_1/bias/v/Read/ReadVariableOp5training_6/Adam/conv2d_2/kernel/v/Read/ReadVariableOp3training_6/Adam/conv2d_2/bias/v/Read/ReadVariableOp=training_6/Adam/variational_mean/kernel/v/Read/ReadVariableOp;training_6/Adam/variational_mean/bias/v/Read/ReadVariableOpEtraining_6/Adam/variational_log_variance/kernel/v/Read/ReadVariableOpCtraining_6/Adam/variational_log_variance/bias/v/Read/ReadVariableOp=training_6/Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp;training_6/Adam/conv2d_transpose/bias/v/Read/ReadVariableOp?training_6/Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp=training_6/Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp?training_6/Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp=training_6/Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOp?training_6/Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp=training_6/Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_9972

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametraining_6/Adam/itertraining_6/Adam/beta_1training_6/Adam/beta_2training_6/Adam/decaytraining_6/Adam/learning_rateconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasvariational_mean/kernelvariational_mean/biasvariational_log_variance/kernelvariational_log_variance/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biastraining_6/Adam/conv2d/kernel/mtraining_6/Adam/conv2d/bias/m!training_6/Adam/conv2d_1/kernel/mtraining_6/Adam/conv2d_1/bias/m!training_6/Adam/conv2d_2/kernel/mtraining_6/Adam/conv2d_2/bias/m)training_6/Adam/variational_mean/kernel/m'training_6/Adam/variational_mean/bias/m1training_6/Adam/variational_log_variance/kernel/m/training_6/Adam/variational_log_variance/bias/m)training_6/Adam/conv2d_transpose/kernel/m'training_6/Adam/conv2d_transpose/bias/m+training_6/Adam/conv2d_transpose_1/kernel/m)training_6/Adam/conv2d_transpose_1/bias/m+training_6/Adam/conv2d_transpose_2/kernel/m)training_6/Adam/conv2d_transpose_2/bias/m+training_6/Adam/conv2d_transpose_3/kernel/m)training_6/Adam/conv2d_transpose_3/bias/mtraining_6/Adam/conv2d/kernel/vtraining_6/Adam/conv2d/bias/v!training_6/Adam/conv2d_1/kernel/vtraining_6/Adam/conv2d_1/bias/v!training_6/Adam/conv2d_2/kernel/vtraining_6/Adam/conv2d_2/bias/v)training_6/Adam/variational_mean/kernel/v'training_6/Adam/variational_mean/bias/v1training_6/Adam/variational_log_variance/kernel/v/training_6/Adam/variational_log_variance/bias/v)training_6/Adam/conv2d_transpose/kernel/v'training_6/Adam/conv2d_transpose/bias/v+training_6/Adam/conv2d_transpose_1/kernel/v)training_6/Adam/conv2d_transpose_1/bias/v+training_6/Adam/conv2d_transpose_2/kernel/v)training_6/Adam/conv2d_transpose_2/bias/v+training_6/Adam/conv2d_transpose_3/kernel/v)training_6/Adam/conv2d_transpose_3/bias/v*G
Tin@
>2<*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_10159Þ
¬
J
.__inference_up_sampling2d_1_layer_call_fn_8174

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_81712
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

B
&__inference_reshape_layer_call_fn_9772

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_83932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

¦
1__inference_conv2d_transpose_2_layer_call_fn_8259

inputs
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_kernelconv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_82542
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
à¢

F__inference_functional_5_layer_call_and_return_conditional_losses_9126

inputs;
7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel:
6functional_1_conv2d_biasadd_readvariableop_conv2d_bias?
;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel>
:functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias?
;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel>
:functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_biasO
Kfunctional_1_variational_mean_matmul_readvariableop_variational_mean_kernelN
Jfunctional_1_variational_mean_biasadd_readvariableop_variational_mean_bias_
[functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel^
Zfunctional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_biasY
Ufunctional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelN
Jfunctional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias]
Yfunctional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelR
Nfunctional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias]
Yfunctional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelR
Nfunctional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias]
Yfunctional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelR
Nfunctional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityÖ
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOpà
functional_1/conv2d/Conv2DConv2Dinputs1functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2
functional_1/conv2d/Conv2DË
*functional_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp6functional_1_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02,
*functional_1/conv2d/BiasAdd/ReadVariableOpØ
functional_1/conv2d/BiasAddBiasAdd#functional_1/conv2d/Conv2D:output:02functional_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
functional_1/conv2d/BiasAdd
functional_1/conv2d/ReluRelu$functional_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
functional_1/conv2d/Reluè
"functional_1/max_pooling2d/MaxPoolMaxPool&functional_1/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2$
"functional_1/max_pooling2d/MaxPoolÞ
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOp
functional_1/conv2d_1/Conv2DConv2D+functional_1/max_pooling2d/MaxPool:output:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
functional_1/conv2d_1/Conv2DÓ
,functional_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02.
,functional_1/conv2d_1/BiasAdd/ReadVariableOpà
functional_1/conv2d_1/BiasAddBiasAdd%functional_1/conv2d_1/Conv2D:output:04functional_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_1/conv2d_1/BiasAdd¢
functional_1/conv2d_1/ReluRelu&functional_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_1/conv2d_1/Reluî
$functional_1/max_pooling2d_1/MaxPoolMaxPool(functional_1/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_1/MaxPoolÞ
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOp
functional_1/conv2d_2/Conv2DConv2D-functional_1/max_pooling2d_1/MaxPool:output:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
functional_1/conv2d_2/Conv2DÓ
,functional_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02.
,functional_1/conv2d_2/BiasAdd/ReadVariableOpà
functional_1/conv2d_2/BiasAddBiasAdd%functional_1/conv2d_2/Conv2D:output:04functional_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_1/conv2d_2/BiasAdd¢
functional_1/conv2d_2/ReluRelu&functional_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_1/conv2d_2/Reluî
$functional_1/max_pooling2d_2/MaxPoolMaxPool(functional_1/conv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_2/MaxPool
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
functional_1/flatten/ConstÎ
functional_1/flatten/ReshapeReshape-functional_1/max_pooling2d_2/MaxPool:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_1/flatten/Reshape÷
3functional_1/variational_mean/MatMul/ReadVariableOpReadVariableOpKfunctional_1_variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype025
3functional_1/variational_mean/MatMul/ReadVariableOpì
$functional_1/variational_mean/MatMulMatMul%functional_1/flatten/Reshape:output:0;functional_1/variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$functional_1/variational_mean/MatMuló
4functional_1/variational_mean/BiasAdd/ReadVariableOpReadVariableOpJfunctional_1_variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype026
4functional_1/variational_mean/BiasAdd/ReadVariableOpù
%functional_1/variational_mean/BiasAddBiasAdd.functional_1/variational_mean/MatMul:product:0<functional_1/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2'
%functional_1/variational_mean/BiasAdd
;functional_1/variational_log_variance/MatMul/ReadVariableOpReadVariableOp[functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype02=
;functional_1/variational_log_variance/MatMul/ReadVariableOp
,functional_1/variational_log_variance/MatMulMatMul%functional_1/flatten/Reshape:output:0Cfunctional_1/variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2.
,functional_1/variational_log_variance/MatMul
<functional_1/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpZfunctional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02>
<functional_1/variational_log_variance/BiasAdd/ReadVariableOp
-functional_1/variational_log_variance/BiasAddBiasAdd6functional_1/variational_log_variance/MatMul:product:0Dfunctional_1/variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2/
-functional_1/variational_log_variance/BiasAdd
functional_1/lambda/ShapeShape6functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_1/lambda/Shape
'functional_1/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'functional_1/lambda/strided_slice/stack 
)functional_1/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice/stack_1 
)functional_1/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice/stack_2Ú
!functional_1/lambda/strided_sliceStridedSlice"functional_1/lambda/Shape:output:00functional_1/lambda/strided_slice/stack:output:02functional_1/lambda/strided_slice/stack_1:output:02functional_1/lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!functional_1/lambda/strided_slice 
functional_1/lambda/Shape_1Shape6functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_1/lambda/Shape_1 
)functional_1/lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice_1/stack¤
+functional_1/lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/lambda/strided_slice_1/stack_1¤
+functional_1/lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/lambda/strided_slice_1/stack_2æ
#functional_1/lambda/strided_slice_1StridedSlice$functional_1/lambda/Shape_1:output:02functional_1/lambda/strided_slice_1/stack:output:04functional_1/lambda/strided_slice_1/stack_1:output:04functional_1/lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#functional_1/lambda/strided_slice_1â
'functional_1/lambda/random_normal/shapePack*functional_1/lambda/strided_slice:output:0,functional_1/lambda/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/lambda/random_normal/shape
&functional_1/lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&functional_1/lambda/random_normal/mean
(functional_1/lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(functional_1/lambda/random_normal/stddev¡
6functional_1/lambda/random_normal/RandomStandardNormalRandomStandardNormal0functional_1/lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2êÁÒ28
6functional_1/lambda/random_normal/RandomStandardNormal
%functional_1/lambda/random_normal/mulMul?functional_1/lambda/random_normal/RandomStandardNormal:output:01functional_1/lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%functional_1/lambda/random_normal/mulä
!functional_1/lambda/random_normalAdd)functional_1/lambda/random_normal/mul:z:0/functional_1/lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!functional_1/lambda/random_normal{
functional_1/lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
functional_1/lambda/mul/xÇ
functional_1/lambda/mulMul"functional_1/lambda/mul/x:output:06functional_1/variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/mul
functional_1/lambda/ExpExpfunctional_1/lambda/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/Exp³
functional_1/lambda/mul_1Mulfunctional_1/lambda/Exp:y:0%functional_1/lambda/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/mul_1¼
functional_1/lambda/addAddV2.functional_1/variational_mean/BiasAdd:output:0functional_1/lambda/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/add
functional_3/reshape/ShapeShapefunctional_1/lambda/add:z:0*
T0*
_output_shapes
:2
functional_3/reshape/Shape
(functional_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_3/reshape/strided_slice/stack¢
*functional_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_3/reshape/strided_slice/stack_1¢
*functional_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_3/reshape/strided_slice/stack_2à
"functional_3/reshape/strided_sliceStridedSlice#functional_3/reshape/Shape:output:01functional_3/reshape/strided_slice/stack:output:03functional_3/reshape/strided_slice/stack_1:output:03functional_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"functional_3/reshape/strided_slice
$functional_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/reshape/Reshape/shape/1
$functional_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/reshape/Reshape/shape/2
$functional_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2&
$functional_3/reshape/Reshape/shape/3¸
"functional_3/reshape/Reshape/shapePack+functional_3/reshape/strided_slice:output:0-functional_3/reshape/Reshape/shape/1:output:0-functional_3/reshape/Reshape/shape/2:output:0-functional_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"functional_3/reshape/Reshape/shapeË
functional_3/reshape/ReshapeReshapefunctional_1/lambda/add:z:0+functional_3/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_3/reshape/Reshape
#functional_3/conv2d_transpose/ShapeShape%functional_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#functional_3/conv2d_transpose/Shape°
1functional_3/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1functional_3/conv2d_transpose/strided_slice/stack´
3functional_3/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_3/conv2d_transpose/strided_slice/stack_1´
3functional_3/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_3/conv2d_transpose/strided_slice/stack_2
+functional_3/conv2d_transpose/strided_sliceStridedSlice,functional_3/conv2d_transpose/Shape:output:0:functional_3/conv2d_transpose/strided_slice/stack:output:0<functional_3/conv2d_transpose/strided_slice/stack_1:output:0<functional_3/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+functional_3/conv2d_transpose/strided_slice
%functional_3/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_3/conv2d_transpose/stack/1
%functional_3/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_3/conv2d_transpose/stack/2
%functional_3/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%functional_3/conv2d_transpose/stack/3Æ
#functional_3/conv2d_transpose/stackPack4functional_3/conv2d_transpose/strided_slice:output:0.functional_3/conv2d_transpose/stack/1:output:0.functional_3/conv2d_transpose/stack/2:output:0.functional_3/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#functional_3/conv2d_transpose/stack´
3functional_3/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose/strided_slice_1/stack¸
5functional_3/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose/strided_slice_1/stack_1¸
5functional_3/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose/strided_slice_1/stack_2 
-functional_3/conv2d_transpose/strided_slice_1StridedSlice,functional_3/conv2d_transpose/stack:output:0<functional_3/conv2d_transpose/strided_slice_1/stack:output:0>functional_3/conv2d_transpose/strided_slice_1/stack_1:output:0>functional_3/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose/strided_slice_1
=functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpUfunctional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02?
=functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpö
.functional_3/conv2d_transpose/conv2d_transposeConv2DBackpropInput,functional_3/conv2d_transpose/stack:output:0Efunctional_3/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%functional_3/reshape/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
20
.functional_3/conv2d_transpose/conv2d_transposeó
4functional_3/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpJfunctional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype026
4functional_3/conv2d_transpose/BiasAdd/ReadVariableOp
%functional_3/conv2d_transpose/BiasAddBiasAdd7functional_3/conv2d_transpose/conv2d_transpose:output:0<functional_3/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%functional_3/conv2d_transpose/BiasAddº
"functional_3/conv2d_transpose/ReluRelu.functional_3/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"functional_3/conv2d_transpose/Relu¤
 functional_3/up_sampling2d/ShapeShape0functional_3/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d/Shapeª
.functional_3/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.functional_3/up_sampling2d/strided_slice/stack®
0functional_3/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d/strided_slice/stack_1®
0functional_3/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d/strided_slice/stack_2ð
(functional_3/up_sampling2d/strided_sliceStridedSlice)functional_3/up_sampling2d/Shape:output:07functional_3/up_sampling2d/strided_slice/stack:output:09functional_3/up_sampling2d/strided_slice/stack_1:output:09functional_3/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(functional_3/up_sampling2d/strided_slice
 functional_3/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2"
 functional_3/up_sampling2d/ConstÊ
functional_3/up_sampling2d/mulMul1functional_3/up_sampling2d/strided_slice:output:0)functional_3/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2 
functional_3/up_sampling2d/mulµ
7functional_3/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor0functional_3/conv2d_transpose/Relu:activations:0"functional_3/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(29
7functional_3/up_sampling2d/resize/ResizeNearestNeighborÆ
%functional_3/conv2d_transpose_1/ShapeShapeHfunctional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_1/Shape´
3functional_3/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_1/strided_slice/stack¸
5functional_3/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_1/strided_slice/stack_1¸
5functional_3/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_1/strided_slice/stack_2¢
-functional_3/conv2d_transpose_1/strided_sliceStridedSlice.functional_3/conv2d_transpose_1/Shape:output:0<functional_3/conv2d_transpose_1/strided_slice/stack:output:0>functional_3/conv2d_transpose_1/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_1/strided_slice
'functional_3/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_1/stack/1
'functional_3/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_1/stack/2
'functional_3/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'functional_3/conv2d_transpose_1/stack/3Ò
%functional_3/conv2d_transpose_1/stackPack6functional_3/conv2d_transpose_1/strided_slice:output:00functional_3/conv2d_transpose_1/stack/1:output:00functional_3/conv2d_transpose_1/stack/2:output:00functional_3/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_1/stack¸
5functional_3/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_1/strided_slice_1/stack¼
7functional_3/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_1/strided_slice_1/stack_1¼
7functional_3/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_1/strided_slice_1/stack_2¬
/functional_3/conv2d_transpose_1/strided_slice_1StridedSlice.functional_3/conv2d_transpose_1/stack:output:0>functional_3/conv2d_transpose_1/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_1/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_1/strided_slice_1¤
?functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02A
?functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOp¡
0functional_3/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_1/stack:output:0Gfunctional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Hfunctional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
22
0functional_3/conv2d_transpose_1/conv2d_transposeû
6functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype028
6functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOp
'functional_3/conv2d_transpose_1/BiasAddBiasAdd9functional_3/conv2d_transpose_1/conv2d_transpose:output:0>functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'functional_3/conv2d_transpose_1/BiasAddÀ
$functional_3/conv2d_transpose_1/ReluRelu0functional_3/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$functional_3/conv2d_transpose_1/Reluª
"functional_3/up_sampling2d_1/ShapeShape2functional_3/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2$
"functional_3/up_sampling2d_1/Shape®
0functional_3/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d_1/strided_slice/stack²
2functional_3/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_1/strided_slice/stack_1²
2functional_3/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_1/strided_slice/stack_2ü
*functional_3/up_sampling2d_1/strided_sliceStridedSlice+functional_3/up_sampling2d_1/Shape:output:09functional_3/up_sampling2d_1/strided_slice/stack:output:0;functional_3/up_sampling2d_1/strided_slice/stack_1:output:0;functional_3/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_3/up_sampling2d_1/strided_slice
"functional_3/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_3/up_sampling2d_1/ConstÒ
 functional_3/up_sampling2d_1/mulMul3functional_3/up_sampling2d_1/strided_slice:output:0+functional_3/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d_1/mul½
9functional_3/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor2functional_3/conv2d_transpose_1/Relu:activations:0$functional_3/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2;
9functional_3/up_sampling2d_1/resize/ResizeNearestNeighborÈ
%functional_3/conv2d_transpose_2/ShapeShapeJfunctional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_2/Shape´
3functional_3/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_2/strided_slice/stack¸
5functional_3/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_2/strided_slice/stack_1¸
5functional_3/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_2/strided_slice/stack_2¢
-functional_3/conv2d_transpose_2/strided_sliceStridedSlice.functional_3/conv2d_transpose_2/Shape:output:0<functional_3/conv2d_transpose_2/strided_slice/stack:output:0>functional_3/conv2d_transpose_2/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_2/strided_slice
'functional_3/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/1
'functional_3/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/2
'functional_3/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/3Ò
%functional_3/conv2d_transpose_2/stackPack6functional_3/conv2d_transpose_2/strided_slice:output:00functional_3/conv2d_transpose_2/stack/1:output:00functional_3/conv2d_transpose_2/stack/2:output:00functional_3/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_2/stack¸
5functional_3/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_2/strided_slice_1/stack¼
7functional_3/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_2/strided_slice_1/stack_1¼
7functional_3/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_2/strided_slice_1/stack_2¬
/functional_3/conv2d_transpose_2/strided_slice_1StridedSlice.functional_3/conv2d_transpose_2/stack:output:0>functional_3/conv2d_transpose_2/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_2/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_2/strided_slice_1¤
?functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02A
?functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOp£
0functional_3/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_2/stack:output:0Gfunctional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0Jfunctional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
22
0functional_3/conv2d_transpose_2/conv2d_transposeû
6functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype028
6functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOp
'functional_3/conv2d_transpose_2/BiasAddBiasAdd9functional_3/conv2d_transpose_2/conv2d_transpose:output:0>functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_3/conv2d_transpose_2/BiasAddÀ
$functional_3/conv2d_transpose_2/ReluRelu0functional_3/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$functional_3/conv2d_transpose_2/Reluª
"functional_3/up_sampling2d_2/ShapeShape2functional_3/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2$
"functional_3/up_sampling2d_2/Shape®
0functional_3/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d_2/strided_slice/stack²
2functional_3/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_2/strided_slice/stack_1²
2functional_3/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_2/strided_slice/stack_2ü
*functional_3/up_sampling2d_2/strided_sliceStridedSlice+functional_3/up_sampling2d_2/Shape:output:09functional_3/up_sampling2d_2/strided_slice/stack:output:0;functional_3/up_sampling2d_2/strided_slice/stack_1:output:0;functional_3/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_3/up_sampling2d_2/strided_slice
"functional_3/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_3/up_sampling2d_2/ConstÒ
 functional_3/up_sampling2d_2/mulMul3functional_3/up_sampling2d_2/strided_slice:output:0+functional_3/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d_2/mul½
9functional_3/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor2functional_3/conv2d_transpose_2/Relu:activations:0$functional_3/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((*
half_pixel_centers(2;
9functional_3/up_sampling2d_2/resize/ResizeNearestNeighborÈ
%functional_3/conv2d_transpose_3/ShapeShapeJfunctional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_3/Shape´
3functional_3/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_3/strided_slice/stack¸
5functional_3/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_3/strided_slice/stack_1¸
5functional_3/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_3/strided_slice/stack_2¢
-functional_3/conv2d_transpose_3/strided_sliceStridedSlice.functional_3/conv2d_transpose_3/Shape:output:0<functional_3/conv2d_transpose_3/strided_slice/stack:output:0>functional_3/conv2d_transpose_3/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_3/strided_slice
'functional_3/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :-2)
'functional_3/conv2d_transpose_3/stack/1
'functional_3/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2)
'functional_3/conv2d_transpose_3/stack/2
'functional_3/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_3/stack/3Ò
%functional_3/conv2d_transpose_3/stackPack6functional_3/conv2d_transpose_3/strided_slice:output:00functional_3/conv2d_transpose_3/stack/1:output:00functional_3/conv2d_transpose_3/stack/2:output:00functional_3/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_3/stack¸
5functional_3/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_3/strided_slice_1/stack¼
7functional_3/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_3/strided_slice_1/stack_1¼
7functional_3/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_3/strided_slice_1/stack_2¬
/functional_3/conv2d_transpose_3/strided_slice_1StridedSlice.functional_3/conv2d_transpose_3/stack:output:0>functional_3/conv2d_transpose_3/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_3/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_3/strided_slice_1¤
?functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02A
?functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOp£
0functional_3/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_3/stack:output:0Gfunctional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0Jfunctional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*
paddingVALID*
strides
22
0functional_3/conv2d_transpose_3/conv2d_transposeû
6functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype028
6functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOp
'functional_3/conv2d_transpose_3/BiasAddBiasAdd9functional_3/conv2d_transpose_3/conv2d_transpose:output:0>functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2)
'functional_3/conv2d_transpose_3/BiasAddÀ
$functional_3/conv2d_transpose_3/ReluRelu0functional_3/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2&
$functional_3/conv2d_transpose_3/Relu
IdentityIdentity2functional_3/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--:::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
ó¦
Á
F__inference_functional_3_layer_call_and_return_conditional_losses_9564

inputsL
Hconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelA
=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_biasP
Lconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelE
Aconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_biasP
Lconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelE
Aconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_biasP
Lconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelE
Aconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shape/3ê
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2È
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
conv2d_transpose/stack/3ø
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2Ò
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1õ
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpµ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transposeÌ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOpÖ
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_transpose/BiasAdd
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_transpose/Relu}
up_sampling2d/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2¢
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
up_sampling2d/Const
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#conv2d_transpose/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor
conv2d_transpose_1/ShapeShape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2Ô
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
conv2d_transpose_1/stack/3
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack¢
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1¢
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2Þ
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1ý
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpà
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transposeÔ
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpÞ
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_transpose_1/BiasAdd
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_transpose_1/Relu
up_sampling2d_1/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2®
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
up_sampling2d_1/Const
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_1/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor¡
conv2d_transpose_2/ShapeShape=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2Ô
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
conv2d_transpose_2/stack/3
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack¢
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1¢
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2Þ
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1ý
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpâ
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transposeÔ
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpÞ
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_transpose_2/BiasAdd
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_transpose_2/Relu
up_sampling2d_2/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2®
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
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_2/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor¡
conv2d_transpose_3/ShapeShape=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2Ô
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
conv2d_transpose_3/stack/3
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack¢
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1¢
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2Þ
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1ý
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpâ
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transposeÔ
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpÞ
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2
conv2d_transpose_3/BiasAdd
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2
conv2d_transpose_3/Relu
IdentityIdentity%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd:::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

o
@__inference_lambda_layer_call_and_return_conditional_losses_9741
inputs_0
inputs_1
identityF
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
strided_slice/stack_2â
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
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
 *  ?2
random_normal/stddevä
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¸Å~2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿd2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/1
ÿ
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7602

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7619

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
&
Þ
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8370

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel2
.biasadd_readvariableop_conv2d_transpose_3_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Ä
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8171

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

Ø
+__inference_functional_1_layer_call_fn_9313

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
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_78902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
*

F__inference_functional_3_layer_call_and_return_conditional_losses_8460
input_2,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identity¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCallÛ
reshape/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_83932
reshape/PartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_80222*
(conv2d_transpose/StatefulPartitionedCall©
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_80552
up_sampling2d/PartitionedCall¥
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_81382,
*conv2d_transpose_1/StatefulPartitionedCall±
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_81712!
up_sampling2d_1/PartitionedCall§
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_82542,
*conv2d_transpose_2/StatefulPartitionedCall±
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_82872!
up_sampling2d_2/PartitionedCall§
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_83702,
*conv2d_transpose_3/StatefulPartitionedCallÓ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_2
Ç

Ñ
+__inference_functional_3_layer_call_fn_8547
input_2
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85362
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_2
·

Ù
+__inference_functional_1_layer_call_fn_7942
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
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_79292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1

m
@__inference_lambda_layer_call_and_return_conditional_losses_7797

inputs
inputs_1
identityF
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
strided_slice/stack_2â
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
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
 *  ?2
random_normal/stddevå
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¡ÛÃ2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿd2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ß¢

F__inference_functional_5_layer_call_and_return_conditional_losses_8949

inputs;
7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel:
6functional_1_conv2d_biasadd_readvariableop_conv2d_bias?
;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel>
:functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias?
;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel>
:functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_biasO
Kfunctional_1_variational_mean_matmul_readvariableop_variational_mean_kernelN
Jfunctional_1_variational_mean_biasadd_readvariableop_variational_mean_bias_
[functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel^
Zfunctional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_biasY
Ufunctional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelN
Jfunctional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias]
Yfunctional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelR
Nfunctional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias]
Yfunctional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelR
Nfunctional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias]
Yfunctional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelR
Nfunctional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityÖ
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOpà
functional_1/conv2d/Conv2DConv2Dinputs1functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2
functional_1/conv2d/Conv2DË
*functional_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp6functional_1_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02,
*functional_1/conv2d/BiasAdd/ReadVariableOpØ
functional_1/conv2d/BiasAddBiasAdd#functional_1/conv2d/Conv2D:output:02functional_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
functional_1/conv2d/BiasAdd
functional_1/conv2d/ReluRelu$functional_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
functional_1/conv2d/Reluè
"functional_1/max_pooling2d/MaxPoolMaxPool&functional_1/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2$
"functional_1/max_pooling2d/MaxPoolÞ
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOp
functional_1/conv2d_1/Conv2DConv2D+functional_1/max_pooling2d/MaxPool:output:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
functional_1/conv2d_1/Conv2DÓ
,functional_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02.
,functional_1/conv2d_1/BiasAdd/ReadVariableOpà
functional_1/conv2d_1/BiasAddBiasAdd%functional_1/conv2d_1/Conv2D:output:04functional_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_1/conv2d_1/BiasAdd¢
functional_1/conv2d_1/ReluRelu&functional_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_1/conv2d_1/Reluî
$functional_1/max_pooling2d_1/MaxPoolMaxPool(functional_1/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_1/MaxPoolÞ
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOp
functional_1/conv2d_2/Conv2DConv2D-functional_1/max_pooling2d_1/MaxPool:output:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
functional_1/conv2d_2/Conv2DÓ
,functional_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02.
,functional_1/conv2d_2/BiasAdd/ReadVariableOpà
functional_1/conv2d_2/BiasAddBiasAdd%functional_1/conv2d_2/Conv2D:output:04functional_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_1/conv2d_2/BiasAdd¢
functional_1/conv2d_2/ReluRelu&functional_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_1/conv2d_2/Reluî
$functional_1/max_pooling2d_2/MaxPoolMaxPool(functional_1/conv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_2/MaxPool
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
functional_1/flatten/ConstÎ
functional_1/flatten/ReshapeReshape-functional_1/max_pooling2d_2/MaxPool:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
functional_1/flatten/Reshape÷
3functional_1/variational_mean/MatMul/ReadVariableOpReadVariableOpKfunctional_1_variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype025
3functional_1/variational_mean/MatMul/ReadVariableOpì
$functional_1/variational_mean/MatMulMatMul%functional_1/flatten/Reshape:output:0;functional_1/variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$functional_1/variational_mean/MatMuló
4functional_1/variational_mean/BiasAdd/ReadVariableOpReadVariableOpJfunctional_1_variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype026
4functional_1/variational_mean/BiasAdd/ReadVariableOpù
%functional_1/variational_mean/BiasAddBiasAdd.functional_1/variational_mean/MatMul:product:0<functional_1/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2'
%functional_1/variational_mean/BiasAdd
;functional_1/variational_log_variance/MatMul/ReadVariableOpReadVariableOp[functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype02=
;functional_1/variational_log_variance/MatMul/ReadVariableOp
,functional_1/variational_log_variance/MatMulMatMul%functional_1/flatten/Reshape:output:0Cfunctional_1/variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2.
,functional_1/variational_log_variance/MatMul
<functional_1/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpZfunctional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02>
<functional_1/variational_log_variance/BiasAdd/ReadVariableOp
-functional_1/variational_log_variance/BiasAddBiasAdd6functional_1/variational_log_variance/MatMul:product:0Dfunctional_1/variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2/
-functional_1/variational_log_variance/BiasAdd
functional_1/lambda/ShapeShape6functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_1/lambda/Shape
'functional_1/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'functional_1/lambda/strided_slice/stack 
)functional_1/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice/stack_1 
)functional_1/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice/stack_2Ú
!functional_1/lambda/strided_sliceStridedSlice"functional_1/lambda/Shape:output:00functional_1/lambda/strided_slice/stack:output:02functional_1/lambda/strided_slice/stack_1:output:02functional_1/lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!functional_1/lambda/strided_slice 
functional_1/lambda/Shape_1Shape6functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_1/lambda/Shape_1 
)functional_1/lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)functional_1/lambda/strided_slice_1/stack¤
+functional_1/lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/lambda/strided_slice_1/stack_1¤
+functional_1/lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/lambda/strided_slice_1/stack_2æ
#functional_1/lambda/strided_slice_1StridedSlice$functional_1/lambda/Shape_1:output:02functional_1/lambda/strided_slice_1/stack:output:04functional_1/lambda/strided_slice_1/stack_1:output:04functional_1/lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#functional_1/lambda/strided_slice_1â
'functional_1/lambda/random_normal/shapePack*functional_1/lambda/strided_slice:output:0,functional_1/lambda/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/lambda/random_normal/shape
&functional_1/lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&functional_1/lambda/random_normal/mean
(functional_1/lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(functional_1/lambda/random_normal/stddev 
6functional_1/lambda/random_normal/RandomStandardNormalRandomStandardNormal0functional_1/lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2É428
6functional_1/lambda/random_normal/RandomStandardNormal
%functional_1/lambda/random_normal/mulMul?functional_1/lambda/random_normal/RandomStandardNormal:output:01functional_1/lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%functional_1/lambda/random_normal/mulä
!functional_1/lambda/random_normalAdd)functional_1/lambda/random_normal/mul:z:0/functional_1/lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!functional_1/lambda/random_normal{
functional_1/lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
functional_1/lambda/mul/xÇ
functional_1/lambda/mulMul"functional_1/lambda/mul/x:output:06functional_1/variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/mul
functional_1/lambda/ExpExpfunctional_1/lambda/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/Exp³
functional_1/lambda/mul_1Mulfunctional_1/lambda/Exp:y:0%functional_1/lambda/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/mul_1¼
functional_1/lambda/addAddV2.functional_1/variational_mean/BiasAdd:output:0functional_1/lambda/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_1/lambda/add
functional_3/reshape/ShapeShapefunctional_1/lambda/add:z:0*
T0*
_output_shapes
:2
functional_3/reshape/Shape
(functional_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_3/reshape/strided_slice/stack¢
*functional_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_3/reshape/strided_slice/stack_1¢
*functional_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_3/reshape/strided_slice/stack_2à
"functional_3/reshape/strided_sliceStridedSlice#functional_3/reshape/Shape:output:01functional_3/reshape/strided_slice/stack:output:03functional_3/reshape/strided_slice/stack_1:output:03functional_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"functional_3/reshape/strided_slice
$functional_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/reshape/Reshape/shape/1
$functional_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$functional_3/reshape/Reshape/shape/2
$functional_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d2&
$functional_3/reshape/Reshape/shape/3¸
"functional_3/reshape/Reshape/shapePack+functional_3/reshape/strided_slice:output:0-functional_3/reshape/Reshape/shape/1:output:0-functional_3/reshape/Reshape/shape/2:output:0-functional_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"functional_3/reshape/Reshape/shapeË
functional_3/reshape/ReshapeReshapefunctional_1/lambda/add:z:0+functional_3/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_3/reshape/Reshape
#functional_3/conv2d_transpose/ShapeShape%functional_3/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#functional_3/conv2d_transpose/Shape°
1functional_3/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1functional_3/conv2d_transpose/strided_slice/stack´
3functional_3/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_3/conv2d_transpose/strided_slice/stack_1´
3functional_3/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_3/conv2d_transpose/strided_slice/stack_2
+functional_3/conv2d_transpose/strided_sliceStridedSlice,functional_3/conv2d_transpose/Shape:output:0:functional_3/conv2d_transpose/strided_slice/stack:output:0<functional_3/conv2d_transpose/strided_slice/stack_1:output:0<functional_3/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+functional_3/conv2d_transpose/strided_slice
%functional_3/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_3/conv2d_transpose/stack/1
%functional_3/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_3/conv2d_transpose/stack/2
%functional_3/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%functional_3/conv2d_transpose/stack/3Æ
#functional_3/conv2d_transpose/stackPack4functional_3/conv2d_transpose/strided_slice:output:0.functional_3/conv2d_transpose/stack/1:output:0.functional_3/conv2d_transpose/stack/2:output:0.functional_3/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#functional_3/conv2d_transpose/stack´
3functional_3/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose/strided_slice_1/stack¸
5functional_3/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose/strided_slice_1/stack_1¸
5functional_3/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose/strided_slice_1/stack_2 
-functional_3/conv2d_transpose/strided_slice_1StridedSlice,functional_3/conv2d_transpose/stack:output:0<functional_3/conv2d_transpose/strided_slice_1/stack:output:0>functional_3/conv2d_transpose/strided_slice_1/stack_1:output:0>functional_3/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose/strided_slice_1
=functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpUfunctional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02?
=functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpö
.functional_3/conv2d_transpose/conv2d_transposeConv2DBackpropInput,functional_3/conv2d_transpose/stack:output:0Efunctional_3/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%functional_3/reshape/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
20
.functional_3/conv2d_transpose/conv2d_transposeó
4functional_3/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpJfunctional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype026
4functional_3/conv2d_transpose/BiasAdd/ReadVariableOp
%functional_3/conv2d_transpose/BiasAddBiasAdd7functional_3/conv2d_transpose/conv2d_transpose:output:0<functional_3/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%functional_3/conv2d_transpose/BiasAddº
"functional_3/conv2d_transpose/ReluRelu.functional_3/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"functional_3/conv2d_transpose/Relu¤
 functional_3/up_sampling2d/ShapeShape0functional_3/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d/Shapeª
.functional_3/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.functional_3/up_sampling2d/strided_slice/stack®
0functional_3/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d/strided_slice/stack_1®
0functional_3/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d/strided_slice/stack_2ð
(functional_3/up_sampling2d/strided_sliceStridedSlice)functional_3/up_sampling2d/Shape:output:07functional_3/up_sampling2d/strided_slice/stack:output:09functional_3/up_sampling2d/strided_slice/stack_1:output:09functional_3/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(functional_3/up_sampling2d/strided_slice
 functional_3/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2"
 functional_3/up_sampling2d/ConstÊ
functional_3/up_sampling2d/mulMul1functional_3/up_sampling2d/strided_slice:output:0)functional_3/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2 
functional_3/up_sampling2d/mulµ
7functional_3/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor0functional_3/conv2d_transpose/Relu:activations:0"functional_3/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(29
7functional_3/up_sampling2d/resize/ResizeNearestNeighborÆ
%functional_3/conv2d_transpose_1/ShapeShapeHfunctional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_1/Shape´
3functional_3/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_1/strided_slice/stack¸
5functional_3/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_1/strided_slice/stack_1¸
5functional_3/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_1/strided_slice/stack_2¢
-functional_3/conv2d_transpose_1/strided_sliceStridedSlice.functional_3/conv2d_transpose_1/Shape:output:0<functional_3/conv2d_transpose_1/strided_slice/stack:output:0>functional_3/conv2d_transpose_1/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_1/strided_slice
'functional_3/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_1/stack/1
'functional_3/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_1/stack/2
'functional_3/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'functional_3/conv2d_transpose_1/stack/3Ò
%functional_3/conv2d_transpose_1/stackPack6functional_3/conv2d_transpose_1/strided_slice:output:00functional_3/conv2d_transpose_1/stack/1:output:00functional_3/conv2d_transpose_1/stack/2:output:00functional_3/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_1/stack¸
5functional_3/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_1/strided_slice_1/stack¼
7functional_3/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_1/strided_slice_1/stack_1¼
7functional_3/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_1/strided_slice_1/stack_2¬
/functional_3/conv2d_transpose_1/strided_slice_1StridedSlice.functional_3/conv2d_transpose_1/stack:output:0>functional_3/conv2d_transpose_1/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_1/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_1/strided_slice_1¤
?functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02A
?functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOp¡
0functional_3/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_1/stack:output:0Gfunctional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Hfunctional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
22
0functional_3/conv2d_transpose_1/conv2d_transposeû
6functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype028
6functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOp
'functional_3/conv2d_transpose_1/BiasAddBiasAdd9functional_3/conv2d_transpose_1/conv2d_transpose:output:0>functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'functional_3/conv2d_transpose_1/BiasAddÀ
$functional_3/conv2d_transpose_1/ReluRelu0functional_3/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$functional_3/conv2d_transpose_1/Reluª
"functional_3/up_sampling2d_1/ShapeShape2functional_3/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2$
"functional_3/up_sampling2d_1/Shape®
0functional_3/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d_1/strided_slice/stack²
2functional_3/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_1/strided_slice/stack_1²
2functional_3/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_1/strided_slice/stack_2ü
*functional_3/up_sampling2d_1/strided_sliceStridedSlice+functional_3/up_sampling2d_1/Shape:output:09functional_3/up_sampling2d_1/strided_slice/stack:output:0;functional_3/up_sampling2d_1/strided_slice/stack_1:output:0;functional_3/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_3/up_sampling2d_1/strided_slice
"functional_3/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_3/up_sampling2d_1/ConstÒ
 functional_3/up_sampling2d_1/mulMul3functional_3/up_sampling2d_1/strided_slice:output:0+functional_3/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d_1/mul½
9functional_3/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor2functional_3/conv2d_transpose_1/Relu:activations:0$functional_3/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2;
9functional_3/up_sampling2d_1/resize/ResizeNearestNeighborÈ
%functional_3/conv2d_transpose_2/ShapeShapeJfunctional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_2/Shape´
3functional_3/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_2/strided_slice/stack¸
5functional_3/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_2/strided_slice/stack_1¸
5functional_3/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_2/strided_slice/stack_2¢
-functional_3/conv2d_transpose_2/strided_sliceStridedSlice.functional_3/conv2d_transpose_2/Shape:output:0<functional_3/conv2d_transpose_2/strided_slice/stack:output:0>functional_3/conv2d_transpose_2/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_2/strided_slice
'functional_3/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/1
'functional_3/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/2
'functional_3/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_2/stack/3Ò
%functional_3/conv2d_transpose_2/stackPack6functional_3/conv2d_transpose_2/strided_slice:output:00functional_3/conv2d_transpose_2/stack/1:output:00functional_3/conv2d_transpose_2/stack/2:output:00functional_3/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_2/stack¸
5functional_3/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_2/strided_slice_1/stack¼
7functional_3/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_2/strided_slice_1/stack_1¼
7functional_3/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_2/strided_slice_1/stack_2¬
/functional_3/conv2d_transpose_2/strided_slice_1StridedSlice.functional_3/conv2d_transpose_2/stack:output:0>functional_3/conv2d_transpose_2/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_2/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_2/strided_slice_1¤
?functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02A
?functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOp£
0functional_3/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_2/stack:output:0Gfunctional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0Jfunctional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
22
0functional_3/conv2d_transpose_2/conv2d_transposeû
6functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype028
6functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOp
'functional_3/conv2d_transpose_2/BiasAddBiasAdd9functional_3/conv2d_transpose_2/conv2d_transpose:output:0>functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_3/conv2d_transpose_2/BiasAddÀ
$functional_3/conv2d_transpose_2/ReluRelu0functional_3/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$functional_3/conv2d_transpose_2/Reluª
"functional_3/up_sampling2d_2/ShapeShape2functional_3/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2$
"functional_3/up_sampling2d_2/Shape®
0functional_3/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_3/up_sampling2d_2/strided_slice/stack²
2functional_3/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_2/strided_slice/stack_1²
2functional_3/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_3/up_sampling2d_2/strided_slice/stack_2ü
*functional_3/up_sampling2d_2/strided_sliceStridedSlice+functional_3/up_sampling2d_2/Shape:output:09functional_3/up_sampling2d_2/strided_slice/stack:output:0;functional_3/up_sampling2d_2/strided_slice/stack_1:output:0;functional_3/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_3/up_sampling2d_2/strided_slice
"functional_3/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_3/up_sampling2d_2/ConstÒ
 functional_3/up_sampling2d_2/mulMul3functional_3/up_sampling2d_2/strided_slice:output:0+functional_3/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2"
 functional_3/up_sampling2d_2/mul½
9functional_3/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor2functional_3/conv2d_transpose_2/Relu:activations:0$functional_3/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((*
half_pixel_centers(2;
9functional_3/up_sampling2d_2/resize/ResizeNearestNeighborÈ
%functional_3/conv2d_transpose_3/ShapeShapeJfunctional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_3/Shape´
3functional_3/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3functional_3/conv2d_transpose_3/strided_slice/stack¸
5functional_3/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_3/strided_slice/stack_1¸
5functional_3/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5functional_3/conv2d_transpose_3/strided_slice/stack_2¢
-functional_3/conv2d_transpose_3/strided_sliceStridedSlice.functional_3/conv2d_transpose_3/Shape:output:0<functional_3/conv2d_transpose_3/strided_slice/stack:output:0>functional_3/conv2d_transpose_3/strided_slice/stack_1:output:0>functional_3/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-functional_3/conv2d_transpose_3/strided_slice
'functional_3/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :-2)
'functional_3/conv2d_transpose_3/stack/1
'functional_3/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-2)
'functional_3/conv2d_transpose_3/stack/2
'functional_3/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'functional_3/conv2d_transpose_3/stack/3Ò
%functional_3/conv2d_transpose_3/stackPack6functional_3/conv2d_transpose_3/strided_slice:output:00functional_3/conv2d_transpose_3/stack/1:output:00functional_3/conv2d_transpose_3/stack/2:output:00functional_3/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%functional_3/conv2d_transpose_3/stack¸
5functional_3/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_3/conv2d_transpose_3/strided_slice_1/stack¼
7functional_3/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_3/strided_slice_1/stack_1¼
7functional_3/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_3/conv2d_transpose_3/strided_slice_1/stack_2¬
/functional_3/conv2d_transpose_3/strided_slice_1StridedSlice.functional_3/conv2d_transpose_3/stack:output:0>functional_3/conv2d_transpose_3/strided_slice_1/stack:output:0@functional_3/conv2d_transpose_3/strided_slice_1/stack_1:output:0@functional_3/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_3/conv2d_transpose_3/strided_slice_1¤
?functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpYfunctional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02A
?functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOp£
0functional_3/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.functional_3/conv2d_transpose_3/stack:output:0Gfunctional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0Jfunctional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*
paddingVALID*
strides
22
0functional_3/conv2d_transpose_3/conv2d_transposeû
6functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpNfunctional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype028
6functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOp
'functional_3/conv2d_transpose_3/BiasAddBiasAdd9functional_3/conv2d_transpose_3/conv2d_transpose:output:0>functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2)
'functional_3/conv2d_transpose_3/BiasAddÀ
$functional_3/conv2d_transpose_3/ReluRelu0functional_3/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2&
$functional_3/conv2d_transpose_3/Relu
IdentityIdentity2functional_3/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--:::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
Ä

Ð
+__inference_functional_3_layer_call_fn_9577

inputs
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ÿ
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7594

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
]
A__inference_flatten_layer_call_and_return_conditional_losses_7719

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
°
@__inference_conv2d_layer_call_and_return_conditional_losses_7637

inputs'
#conv2d_readvariableop_conv2d_kernel&
"biasadd_readvariableop_conv2d_bias
identity
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ--:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
Þ
þ
F__inference_functional_5_layer_call_and_return_conditional_losses_8652
input_1
functional_1_conv2d_kernel
functional_1_conv2d_bias 
functional_1_conv2d_1_kernel
functional_1_conv2d_1_bias 
functional_1_conv2d_2_kernel
functional_1_conv2d_2_bias(
$functional_1_variational_mean_kernel&
"functional_1_variational_mean_bias0
,functional_1_variational_log_variance_kernel.
*functional_1_variational_log_variance_bias(
$functional_3_conv2d_transpose_kernel&
"functional_3_conv2d_transpose_bias*
&functional_3_conv2d_transpose_1_kernel(
$functional_3_conv2d_transpose_1_bias*
&functional_3_conv2d_transpose_2_kernel(
$functional_3_conv2d_transpose_2_bias*
&functional_3_conv2d_transpose_3_kernel(
$functional_3_conv2d_transpose_3_bias
identity¢$functional_1/StatefulPartitionedCall¢$functional_3/StatefulPartitionedCallÞ
$functional_1/StatefulPartitionedCallStatefulPartitionedCallinput_1functional_1_conv2d_kernelfunctional_1_conv2d_biasfunctional_1_conv2d_1_kernelfunctional_1_conv2d_1_biasfunctional_1_conv2d_2_kernelfunctional_1_conv2d_2_bias$functional_1_variational_mean_kernel"functional_1_variational_mean_bias,functional_1_variational_log_variance_kernel*functional_1_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_79292&
$functional_1/StatefulPartitionedCall
$functional_3/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0$functional_3_conv2d_transpose_kernel"functional_3_conv2d_transpose_bias&functional_3_conv2d_transpose_1_kernel$functional_3_conv2d_transpose_1_bias&functional_3_conv2d_transpose_2_kernel$functional_3_conv2d_transpose_2_bias&functional_3_conv2d_transpose_3_kernel$functional_3_conv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85362&
$functional_3/StatefulPartitionedCallé
IdentityIdentity-functional_3/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall%^functional_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2L
$functional_3/StatefulPartitionedCall$functional_3/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
ÿ
¼
"__inference_signature_wrapper_8772
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
variational_log_variance_bias
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_biasconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_75712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
 	
¶
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9619

inputs)
%conv2d_readvariableop_conv2d_1_kernel(
$biasadd_readvariableop_conv2d_1_bias
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
&
Þ
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8097

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel2
.biasadd_readvariableop_conv2d_transpose_1_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Ä
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð
Ä
+__inference_functional_5_layer_call_fn_9172

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
variational_log_variance_bias
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_biasconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_87262
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
 	
¶
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7666

inputs)
%conv2d_readvariableop_conv2d_1_kernel(
$biasadd_readvariableop_conv2d_1_bias
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
´

Ø
+__inference_functional_1_layer_call_fn_9328

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
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_79292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
ó¦
Á
F__inference_functional_3_layer_call_and_return_conditional_losses_9446

inputsL
Hconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernelA
=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_biasP
Lconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernelE
Aconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_biasP
Lconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernelE
Aconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_biasP
Lconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernelE
Aconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shape/3ê
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2È
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
conv2d_transpose/stack/3ø
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2Ò
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1õ
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpµ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transposeÌ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOpÖ
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_transpose/BiasAdd
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_transpose/Relu}
up_sampling2d/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2¢
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
up_sampling2d/Const
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#conv2d_transpose/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor
conv2d_transpose_1/ShapeShape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2Ô
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
conv2d_transpose_1/stack/3
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack¢
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1¢
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2Þ
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1ý
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpà
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transposeÔ
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpÞ
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_transpose_1/BiasAdd
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_transpose_1/Relu
up_sampling2d_1/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2®
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
up_sampling2d_1/Const
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_1/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor¡
conv2d_transpose_2/ShapeShape=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2Ô
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
conv2d_transpose_2/stack/3
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack¢
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1¢
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2Þ
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1ý
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpâ
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transposeÔ
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpÞ
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_transpose_2/BiasAdd
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_transpose_2/Relu
up_sampling2d_2/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2®
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
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_2/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor¡
conv2d_transpose_3/ShapeShape=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2Ô
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
conv2d_transpose_3/stack/3
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack¢
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1¢
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2Þ
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1ý
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpâ
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transposeÔ
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpÞ
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2
conv2d_transpose_3/BiasAdd
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2
conv2d_transpose_3/Relu
IdentityIdentity%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd:::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ä

Ð
+__inference_functional_3_layer_call_fn_9590

inputs
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85362
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Þ
þ
F__inference_functional_5_layer_call_and_return_conditional_losses_8628
input_1
functional_1_conv2d_kernel
functional_1_conv2d_bias 
functional_1_conv2d_1_kernel
functional_1_conv2d_1_bias 
functional_1_conv2d_2_kernel
functional_1_conv2d_2_bias(
$functional_1_variational_mean_kernel&
"functional_1_variational_mean_bias0
,functional_1_variational_log_variance_kernel.
*functional_1_variational_log_variance_bias(
$functional_3_conv2d_transpose_kernel&
"functional_3_conv2d_transpose_bias*
&functional_3_conv2d_transpose_1_kernel(
$functional_3_conv2d_transpose_1_bias*
&functional_3_conv2d_transpose_2_kernel(
$functional_3_conv2d_transpose_2_bias*
&functional_3_conv2d_transpose_3_kernel(
$functional_3_conv2d_transpose_3_bias
identity¢$functional_1/StatefulPartitionedCall¢$functional_3/StatefulPartitionedCallÞ
$functional_1/StatefulPartitionedCallStatefulPartitionedCallinput_1functional_1_conv2d_kernelfunctional_1_conv2d_biasfunctional_1_conv2d_1_kernelfunctional_1_conv2d_1_biasfunctional_1_conv2d_2_kernelfunctional_1_conv2d_2_bias$functional_1_variational_mean_kernel"functional_1_variational_mean_bias,functional_1_variational_log_variance_kernel*functional_1_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_78902&
$functional_1/StatefulPartitionedCall
$functional_3/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0$functional_3_conv2d_transpose_kernel"functional_3_conv2d_transpose_bias&functional_3_conv2d_transpose_1_kernel$functional_3_conv2d_transpose_1_bias&functional_3_conv2d_transpose_2_kernel$functional_3_conv2d_transpose_2_bias&functional_3_conv2d_transpose_3_kernel$functional_3_conv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85032&
$functional_3/StatefulPartitionedCallé
IdentityIdentity-functional_3/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall%^functional_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2L
$functional_3/StatefulPartitionedCall$functional_3/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
¢J
À
F__inference_functional_1_layer_call_and_return_conditional_losses_9298

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
identity¯
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2
conv2d/Conv2D¤
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
conv2d/ReluÁ
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool·
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp×
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_1/Conv2D¬
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/ReluÇ
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool·
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOpÙ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_2/Conv2D¬
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/ReluÇ
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flatten/ReshapeÐ
&variational_mean/MatMul/ReadVariableOpReadVariableOp>variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype02(
&variational_mean/MatMul/ReadVariableOp¸
variational_mean/MatMulMatMulflatten/Reshape:output:0.variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
variational_mean/MatMulÌ
'variational_mean/BiasAdd/ReadVariableOpReadVariableOp=variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02)
'variational_mean/BiasAdd/ReadVariableOpÅ
variational_mean/BiasAddBiasAdd!variational_mean/MatMul:product:0/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
variational_mean/BiasAddð
.variational_log_variance/MatMul/ReadVariableOpReadVariableOpNvariational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype020
.variational_log_variance/MatMul/ReadVariableOpÐ
variational_log_variance/MatMulMatMulflatten/Reshape:output:06variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
variational_log_variance/MatMulì
/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpMvariational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype021
/variational_log_variance/BiasAdd/ReadVariableOpå
 variational_log_variance/BiasAddBiasAdd)variational_log_variance/MatMul:product:07variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 variational_log_variance/BiasAddu
lambda/ShapeShape)variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
lambda/Shape
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lambda/strided_slice/stack
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_1
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_2
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
lambda/Shape_1
lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice_1/stack
lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_1
lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_2
lambda/strided_slice_1StridedSlicelambda/Shape_1:output:0%lambda/strided_slice_1/stack:output:0'lambda/strided_slice_1/stack_1:output:0'lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lambda/strided_slice_1®
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
 *  ?2
lambda/random_normal/stddevú
)lambda/random_normal/RandomStandardNormalRandomStandardNormal#lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ûõ2+
)lambda/random_normal/RandomStandardNormalÐ
lambda/random_normal/mulMul2lambda/random_normal/RandomStandardNormal:output:0$lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lambda/random_normal/mul°
lambda/random_normalAddlambda/random_normal/mul:z:0"lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lambda/random_normala
lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda/mul/x

lambda/mulMullambda/mul/x:output:0)variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

lambda/mula

lambda/ExpExplambda/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

lambda/Exp
lambda/mul_1Mullambda/Exp:y:0lambda/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lambda/mul_1

lambda/addAddV2!variational_mean/BiasAdd:output:0lambda/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

lambda/addb
IdentityIdentitylambda/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--:::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
¡J
À
F__inference_functional_1_layer_call_and_return_conditional_losses_9235

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
identity¯
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2
conv2d/Conv2D¤
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
conv2d/ReluÁ
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool·
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp×
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_1/Conv2D¬
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/ReluÇ
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool·
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02 
conv2d_2/Conv2D/ReadVariableOpÙ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_2/Conv2D¬
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/ReluÇ
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flatten/ReshapeÐ
&variational_mean/MatMul/ReadVariableOpReadVariableOp>variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype02(
&variational_mean/MatMul/ReadVariableOp¸
variational_mean/MatMulMatMulflatten/Reshape:output:0.variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
variational_mean/MatMulÌ
'variational_mean/BiasAdd/ReadVariableOpReadVariableOp=variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02)
'variational_mean/BiasAdd/ReadVariableOpÅ
variational_mean/BiasAddBiasAdd!variational_mean/MatMul:product:0/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
variational_mean/BiasAddð
.variational_log_variance/MatMul/ReadVariableOpReadVariableOpNvariational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype020
.variational_log_variance/MatMul/ReadVariableOpÐ
variational_log_variance/MatMulMatMulflatten/Reshape:output:06variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
variational_log_variance/MatMulì
/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpMvariational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype021
/variational_log_variance/BiasAdd/ReadVariableOpå
 variational_log_variance/BiasAddBiasAdd)variational_log_variance/MatMul:product:07variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 variational_log_variance/BiasAddu
lambda/ShapeShape)variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2
lambda/Shape
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lambda/strided_slice/stack
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_1
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice/stack_2
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
lambda/Shape_1
lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
lambda/strided_slice_1/stack
lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_1
lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lambda/strided_slice_1/stack_2
lambda/strided_slice_1StridedSlicelambda/Shape_1:output:0%lambda/strided_slice_1/stack:output:0'lambda/strided_slice_1/stack_1:output:0'lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lambda/strided_slice_1®
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
 *  ?2
lambda/random_normal/stddevù
)lambda/random_normal/RandomStandardNormalRandomStandardNormal#lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¶z2+
)lambda/random_normal/RandomStandardNormalÐ
lambda/random_normal/mulMul2lambda/random_normal/RandomStandardNormal:output:0$lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lambda/random_normal/mul°
lambda/random_normalAddlambda/random_normal/mul:z:0"lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lambda/random_normala
lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lambda/mul/x

lambda/mulMullambda/mul/x:output:0)variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

lambda/mula

lambda/ExpExplambda/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

lambda/Exp
lambda/mul_1Mullambda/Exp:y:0lambda/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lambda/mul_1

lambda/addAddV2!variational_mean/BiasAdd:output:0lambda/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

lambda/addb
IdentityIdentitylambda/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--:::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
	
°
@__inference_conv2d_layer_call_and_return_conditional_losses_9601

inputs'
#conv2d_readvariableop_conv2d_kernel&
"biasadd_readvariableop_conv2d_bias
identity
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ--:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
 	
¶
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7695

inputs)
%conv2d_readvariableop_conv2d_2_kernel(
$biasadd_readvariableop_conv2d_2_bias
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ		@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
ÿ
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7611

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 	
¶
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9637

inputs)
%conv2d_readvariableop_conv2d_2_kernel(
$biasadd_readvariableop_conv2d_2_bias
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ		@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
Á0
Å
F__inference_functional_1_layer_call_and_return_conditional_losses_7890

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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢lambda/StatefulPartitionedCall¢0variational_log_variance/StatefulPartitionedCall¢(variational_mean/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_76372 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_75852
max_pooling2d/PartitionedCallÍ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_76662"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_76022!
max_pooling2d_1/PartitionedCallÏ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_76952"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_76192!
max_pooling2d_2/PartitionedCallõ
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_77192
flatten/PartitionedCall÷
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_variational_mean_layer_call_and_return_conditional_losses_77372*
(variational_mean/StatefulPartitionedCall¯
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_775922
0variational_log_variance/StatefulPartitionedCallÎ
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_77972 
lambda/StatefulPartitionedCallá
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
¬
J
.__inference_up_sampling2d_2_layer_call_fn_8290

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_82872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_conv2d_1_layer_call_fn_9626

inputs
conv2d_1_kernel
conv2d_1_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernelconv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_76662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
*

F__inference_functional_3_layer_call_and_return_conditional_losses_8503

inputs,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identity¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCallÚ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_83932
reshape/PartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_80222*
(conv2d_transpose/StatefulPartitionedCall©
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_80552
up_sampling2d/PartitionedCall¥
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_81382,
*conv2d_transpose_1/StatefulPartitionedCall±
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_81712!
up_sampling2d_1/PartitionedCall§
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_82542,
*conv2d_transpose_2/StatefulPartitionedCall±
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_82872!
up_sampling2d_2/PartitionedCall§
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_83702,
*conv2d_transpose_3/StatefulPartitionedCallÓ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


%__inference_conv2d_layer_call_fn_9608

inputs
conv2d_kernel
conv2d_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_76372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ--::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
&
Þ
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_8213

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel2
.biasadd_readvariableop_conv2d_transpose_2_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Ä
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Î
J__inference_variational_mean_layer_call_and_return_conditional_losses_7737

inputs1
-matmul_readvariableop_variational_mean_kernel0
,biasadd_readvariableop_variational_mean_bias
identity
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶
æ
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_7759

inputs9
5matmul_readvariableop_variational_log_variance_kernel8
4biasadd_readvariableop_variational_log_variance_bias
identity¥
MatMul/ReadVariableOpReadVariableOp5matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul¡
BiasAdd/ReadVariableOpReadVariableOp4biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8272

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

m
@__inference_lambda_layer_call_and_return_conditional_losses_7823

inputs
inputs_1
identityF
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
strided_slice/stack_2â
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
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
 *  ?2
random_normal/stddevå
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2²Í2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿd2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
&
Þ
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8329

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel2
.biasadd_readvariableop_conv2d_transpose_3_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Ä
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
%
!__inference__traced_restore_10159
file_prefix)
%assignvariableop_training_6_adam_iter-
)assignvariableop_1_training_6_adam_beta_1-
)assignvariableop_2_training_6_adam_beta_2,
(assignvariableop_3_training_6_adam_decay4
0assignvariableop_4_training_6_adam_learning_rate$
 assignvariableop_5_conv2d_kernel"
assignvariableop_6_conv2d_bias&
"assignvariableop_7_conv2d_1_kernel$
 assignvariableop_8_conv2d_1_bias&
"assignvariableop_9_conv2d_2_kernel%
!assignvariableop_10_conv2d_2_bias/
+assignvariableop_11_variational_mean_kernel-
)assignvariableop_12_variational_mean_bias7
3assignvariableop_13_variational_log_variance_kernel5
1assignvariableop_14_variational_log_variance_bias/
+assignvariableop_15_conv2d_transpose_kernel-
)assignvariableop_16_conv2d_transpose_bias1
-assignvariableop_17_conv2d_transpose_1_kernel/
+assignvariableop_18_conv2d_transpose_1_bias1
-assignvariableop_19_conv2d_transpose_2_kernel/
+assignvariableop_20_conv2d_transpose_2_bias1
-assignvariableop_21_conv2d_transpose_3_kernel/
+assignvariableop_22_conv2d_transpose_3_bias7
3assignvariableop_23_training_6_adam_conv2d_kernel_m5
1assignvariableop_24_training_6_adam_conv2d_bias_m9
5assignvariableop_25_training_6_adam_conv2d_1_kernel_m7
3assignvariableop_26_training_6_adam_conv2d_1_bias_m9
5assignvariableop_27_training_6_adam_conv2d_2_kernel_m7
3assignvariableop_28_training_6_adam_conv2d_2_bias_mA
=assignvariableop_29_training_6_adam_variational_mean_kernel_m?
;assignvariableop_30_training_6_adam_variational_mean_bias_mI
Eassignvariableop_31_training_6_adam_variational_log_variance_kernel_mG
Cassignvariableop_32_training_6_adam_variational_log_variance_bias_mA
=assignvariableop_33_training_6_adam_conv2d_transpose_kernel_m?
;assignvariableop_34_training_6_adam_conv2d_transpose_bias_mC
?assignvariableop_35_training_6_adam_conv2d_transpose_1_kernel_mA
=assignvariableop_36_training_6_adam_conv2d_transpose_1_bias_mC
?assignvariableop_37_training_6_adam_conv2d_transpose_2_kernel_mA
=assignvariableop_38_training_6_adam_conv2d_transpose_2_bias_mC
?assignvariableop_39_training_6_adam_conv2d_transpose_3_kernel_mA
=assignvariableop_40_training_6_adam_conv2d_transpose_3_bias_m7
3assignvariableop_41_training_6_adam_conv2d_kernel_v5
1assignvariableop_42_training_6_adam_conv2d_bias_v9
5assignvariableop_43_training_6_adam_conv2d_1_kernel_v7
3assignvariableop_44_training_6_adam_conv2d_1_bias_v9
5assignvariableop_45_training_6_adam_conv2d_2_kernel_v7
3assignvariableop_46_training_6_adam_conv2d_2_bias_vA
=assignvariableop_47_training_6_adam_variational_mean_kernel_v?
;assignvariableop_48_training_6_adam_variational_mean_bias_vI
Eassignvariableop_49_training_6_adam_variational_log_variance_kernel_vG
Cassignvariableop_50_training_6_adam_variational_log_variance_bias_vA
=assignvariableop_51_training_6_adam_conv2d_transpose_kernel_v?
;assignvariableop_52_training_6_adam_conv2d_transpose_bias_vC
?assignvariableop_53_training_6_adam_conv2d_transpose_1_kernel_vA
=assignvariableop_54_training_6_adam_conv2d_transpose_1_bias_vC
?assignvariableop_55_training_6_adam_conv2d_transpose_2_kernel_vA
=assignvariableop_56_training_6_adam_conv2d_transpose_2_bias_vC
?assignvariableop_57_training_6_adam_conv2d_transpose_3_kernel_vA
=assignvariableop_58_training_6_adam_conv2d_transpose_3_bias_v
identity_60¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÚ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesó
ð::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity¤
AssignVariableOpAssignVariableOp%assignvariableop_training_6_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_training_6_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp)assignvariableop_2_training_6_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3­
AssignVariableOp_3AssignVariableOp(assignvariableop_3_training_6_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4µ
AssignVariableOp_4AssignVariableOp0assignvariableop_4_training_6_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¥
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9§
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11³
AssignVariableOp_11AssignVariableOp+assignvariableop_11_variational_mean_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_variational_mean_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13»
AssignVariableOp_13AssignVariableOp3assignvariableop_13_variational_log_variance_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¹
AssignVariableOp_14AssignVariableOp1assignvariableop_14_variational_log_variance_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15³
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_conv2d_transpose_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp-assignvariableop_17_conv2d_transpose_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18³
AssignVariableOp_18AssignVariableOp+assignvariableop_18_conv2d_transpose_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19µ
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_2_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20³
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_conv2d_transpose_3_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22³
AssignVariableOp_22AssignVariableOp+assignvariableop_22_conv2d_transpose_3_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23»
AssignVariableOp_23AssignVariableOp3assignvariableop_23_training_6_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¹
AssignVariableOp_24AssignVariableOp1assignvariableop_24_training_6_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25½
AssignVariableOp_25AssignVariableOp5assignvariableop_25_training_6_adam_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26»
AssignVariableOp_26AssignVariableOp3assignvariableop_26_training_6_adam_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27½
AssignVariableOp_27AssignVariableOp5assignvariableop_27_training_6_adam_conv2d_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28»
AssignVariableOp_28AssignVariableOp3assignvariableop_28_training_6_adam_conv2d_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Å
AssignVariableOp_29AssignVariableOp=assignvariableop_29_training_6_adam_variational_mean_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ã
AssignVariableOp_30AssignVariableOp;assignvariableop_30_training_6_adam_variational_mean_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Í
AssignVariableOp_31AssignVariableOpEassignvariableop_31_training_6_adam_variational_log_variance_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ë
AssignVariableOp_32AssignVariableOpCassignvariableop_32_training_6_adam_variational_log_variance_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Å
AssignVariableOp_33AssignVariableOp=assignvariableop_33_training_6_adam_conv2d_transpose_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ã
AssignVariableOp_34AssignVariableOp;assignvariableop_34_training_6_adam_conv2d_transpose_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ç
AssignVariableOp_35AssignVariableOp?assignvariableop_35_training_6_adam_conv2d_transpose_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Å
AssignVariableOp_36AssignVariableOp=assignvariableop_36_training_6_adam_conv2d_transpose_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ç
AssignVariableOp_37AssignVariableOp?assignvariableop_37_training_6_adam_conv2d_transpose_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Å
AssignVariableOp_38AssignVariableOp=assignvariableop_38_training_6_adam_conv2d_transpose_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ç
AssignVariableOp_39AssignVariableOp?assignvariableop_39_training_6_adam_conv2d_transpose_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Å
AssignVariableOp_40AssignVariableOp=assignvariableop_40_training_6_adam_conv2d_transpose_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41»
AssignVariableOp_41AssignVariableOp3assignvariableop_41_training_6_adam_conv2d_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¹
AssignVariableOp_42AssignVariableOp1assignvariableop_42_training_6_adam_conv2d_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43½
AssignVariableOp_43AssignVariableOp5assignvariableop_43_training_6_adam_conv2d_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44»
AssignVariableOp_44AssignVariableOp3assignvariableop_44_training_6_adam_conv2d_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45½
AssignVariableOp_45AssignVariableOp5assignvariableop_45_training_6_adam_conv2d_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46»
AssignVariableOp_46AssignVariableOp3assignvariableop_46_training_6_adam_conv2d_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Å
AssignVariableOp_47AssignVariableOp=assignvariableop_47_training_6_adam_variational_mean_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ã
AssignVariableOp_48AssignVariableOp;assignvariableop_48_training_6_adam_variational_mean_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Í
AssignVariableOp_49AssignVariableOpEassignvariableop_49_training_6_adam_variational_log_variance_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ë
AssignVariableOp_50AssignVariableOpCassignvariableop_50_training_6_adam_variational_log_variance_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Å
AssignVariableOp_51AssignVariableOp=assignvariableop_51_training_6_adam_conv2d_transpose_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ã
AssignVariableOp_52AssignVariableOp;assignvariableop_52_training_6_adam_conv2d_transpose_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ç
AssignVariableOp_53AssignVariableOp?assignvariableop_53_training_6_adam_conv2d_transpose_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Å
AssignVariableOp_54AssignVariableOp=assignvariableop_54_training_6_adam_conv2d_transpose_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ç
AssignVariableOp_55AssignVariableOp?assignvariableop_55_training_6_adam_conv2d_transpose_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Å
AssignVariableOp_56AssignVariableOp=assignvariableop_56_training_6_adam_conv2d_transpose_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ç
AssignVariableOp_57AssignVariableOp?assignvariableop_57_training_6_adam_conv2d_transpose_3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Å
AssignVariableOp_58AssignVariableOp=assignvariableop_58_training_6_adam_conv2d_transpose_3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpð

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59ã

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*
_input_shapesñ
î: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ð
Ä
+__inference_functional_5_layer_call_fn_9149

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
variational_log_variance_bias
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_biasconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_86792
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
*

F__inference_functional_3_layer_call_and_return_conditional_losses_8536

inputs,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identity¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCallÚ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_83932
reshape/PartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_80222*
(conv2d_transpose/StatefulPartitionedCall©
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_80552
up_sampling2d/PartitionedCall¥
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_81382,
*conv2d_transpose_1/StatefulPartitionedCall±
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_81712!
up_sampling2d_1/PartitionedCall§
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_82542,
*conv2d_transpose_2/StatefulPartitionedCall±
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_82872!
up_sampling2d_2/PartitionedCall§
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_83702,
*conv2d_transpose_3/StatefulPartitionedCallÓ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¹
]
A__inference_flatten_layer_call_and_return_conditional_losses_9650

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ç
Û
__inference__traced_save_9972
file_prefix3
/savev2_training_6_adam_iter_read_readvariableop	5
1savev2_training_6_adam_beta_1_read_readvariableop5
1savev2_training_6_adam_beta_2_read_readvariableop4
0savev2_training_6_adam_decay_read_readvariableop<
8savev2_training_6_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop6
2savev2_variational_mean_kernel_read_readvariableop4
0savev2_variational_mean_bias_read_readvariableop>
:savev2_variational_log_variance_kernel_read_readvariableop<
8savev2_variational_log_variance_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop>
:savev2_training_6_adam_conv2d_kernel_m_read_readvariableop<
8savev2_training_6_adam_conv2d_bias_m_read_readvariableop@
<savev2_training_6_adam_conv2d_1_kernel_m_read_readvariableop>
:savev2_training_6_adam_conv2d_1_bias_m_read_readvariableop@
<savev2_training_6_adam_conv2d_2_kernel_m_read_readvariableop>
:savev2_training_6_adam_conv2d_2_bias_m_read_readvariableopH
Dsavev2_training_6_adam_variational_mean_kernel_m_read_readvariableopF
Bsavev2_training_6_adam_variational_mean_bias_m_read_readvariableopP
Lsavev2_training_6_adam_variational_log_variance_kernel_m_read_readvariableopN
Jsavev2_training_6_adam_variational_log_variance_bias_m_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_kernel_m_read_readvariableopF
Bsavev2_training_6_adam_conv2d_transpose_bias_m_read_readvariableopJ
Fsavev2_training_6_adam_conv2d_transpose_1_kernel_m_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_1_bias_m_read_readvariableopJ
Fsavev2_training_6_adam_conv2d_transpose_2_kernel_m_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_2_bias_m_read_readvariableopJ
Fsavev2_training_6_adam_conv2d_transpose_3_kernel_m_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_3_bias_m_read_readvariableop>
:savev2_training_6_adam_conv2d_kernel_v_read_readvariableop<
8savev2_training_6_adam_conv2d_bias_v_read_readvariableop@
<savev2_training_6_adam_conv2d_1_kernel_v_read_readvariableop>
:savev2_training_6_adam_conv2d_1_bias_v_read_readvariableop@
<savev2_training_6_adam_conv2d_2_kernel_v_read_readvariableop>
:savev2_training_6_adam_conv2d_2_bias_v_read_readvariableopH
Dsavev2_training_6_adam_variational_mean_kernel_v_read_readvariableopF
Bsavev2_training_6_adam_variational_mean_bias_v_read_readvariableopP
Lsavev2_training_6_adam_variational_log_variance_kernel_v_read_readvariableopN
Jsavev2_training_6_adam_variational_log_variance_bias_v_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_kernel_v_read_readvariableopF
Bsavev2_training_6_adam_conv2d_transpose_bias_v_read_readvariableopJ
Fsavev2_training_6_adam_conv2d_transpose_1_kernel_v_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_1_bias_v_read_readvariableopJ
Fsavev2_training_6_adam_conv2d_transpose_2_kernel_v_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_2_bias_v_read_readvariableopJ
Fsavev2_training_6_adam_conv2d_transpose_3_kernel_v_read_readvariableopH
Dsavev2_training_6_adam_conv2d_transpose_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_67e9bb31fb484961873e81260d6b547d/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameþ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesú
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_training_6_adam_iter_read_readvariableop1savev2_training_6_adam_beta_1_read_readvariableop1savev2_training_6_adam_beta_2_read_readvariableop0savev2_training_6_adam_decay_read_readvariableop8savev2_training_6_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop2savev2_variational_mean_kernel_read_readvariableop0savev2_variational_mean_bias_read_readvariableop:savev2_variational_log_variance_kernel_read_readvariableop8savev2_variational_log_variance_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop:savev2_training_6_adam_conv2d_kernel_m_read_readvariableop8savev2_training_6_adam_conv2d_bias_m_read_readvariableop<savev2_training_6_adam_conv2d_1_kernel_m_read_readvariableop:savev2_training_6_adam_conv2d_1_bias_m_read_readvariableop<savev2_training_6_adam_conv2d_2_kernel_m_read_readvariableop:savev2_training_6_adam_conv2d_2_bias_m_read_readvariableopDsavev2_training_6_adam_variational_mean_kernel_m_read_readvariableopBsavev2_training_6_adam_variational_mean_bias_m_read_readvariableopLsavev2_training_6_adam_variational_log_variance_kernel_m_read_readvariableopJsavev2_training_6_adam_variational_log_variance_bias_m_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_kernel_m_read_readvariableopBsavev2_training_6_adam_conv2d_transpose_bias_m_read_readvariableopFsavev2_training_6_adam_conv2d_transpose_1_kernel_m_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_1_bias_m_read_readvariableopFsavev2_training_6_adam_conv2d_transpose_2_kernel_m_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_2_bias_m_read_readvariableopFsavev2_training_6_adam_conv2d_transpose_3_kernel_m_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_3_bias_m_read_readvariableop:savev2_training_6_adam_conv2d_kernel_v_read_readvariableop8savev2_training_6_adam_conv2d_bias_v_read_readvariableop<savev2_training_6_adam_conv2d_1_kernel_v_read_readvariableop:savev2_training_6_adam_conv2d_1_bias_v_read_readvariableop<savev2_training_6_adam_conv2d_2_kernel_v_read_readvariableop:savev2_training_6_adam_conv2d_2_bias_v_read_readvariableopDsavev2_training_6_adam_variational_mean_kernel_v_read_readvariableopBsavev2_training_6_adam_variational_mean_bias_v_read_readvariableopLsavev2_training_6_adam_variational_log_variance_kernel_v_read_readvariableopJsavev2_training_6_adam_variational_log_variance_bias_v_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_kernel_v_read_readvariableopBsavev2_training_6_adam_conv2d_transpose_bias_v_read_readvariableopFsavev2_training_6_adam_conv2d_transpose_1_kernel_v_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_1_bias_v_read_readvariableopFsavev2_training_6_adam_conv2d_transpose_2_kernel_v_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_2_bias_v_read_readvariableopFsavev2_training_6_adam_conv2d_transpose_3_kernel_v_read_readvariableopDsavev2_training_6_adam_conv2d_transpose_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapesï
ì: : : : : : :@:@:@@:@:@ : :	 d:d:	 d:d:@d:@: @: : ::::@:@:@@:@:@ : :	 d:d:	 d:d:@d:@: @: : ::::@:@:@@:@:@ : :	 d:d:	 d:d:@d:@: @: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@ : 

_output_shapes
: :%!

_output_shapes
:	 d: 

_output_shapes
:d:%!

_output_shapes
:	 d: 

_output_shapes
:d:,(
&
_output_shapes
:@d: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :%!

_output_shapes
:	 d: 

_output_shapes
:d:% !

_output_shapes
:	 d: !

_output_shapes
:d:,"(
&
_output_shapes
:@d: #

_output_shapes
:@:,$(
&
_output_shapes
: @: %

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:@: +

_output_shapes
:@:,,(
&
_output_shapes
:@@: -

_output_shapes
:@:,.(
&
_output_shapes
:@ : /

_output_shapes
: :%0!

_output_shapes
:	 d: 1

_output_shapes
:d:%2!

_output_shapes
:	 d: 3

_output_shapes
:d:,4(
&
_output_shapes
:@d: 5

_output_shapes
:@:,6(
&
_output_shapes
: @: 7

_output_shapes
: :,8(
&
_output_shapes
: : 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::<

_output_shapes
: 

e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8156

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·

Ù
+__inference_functional_1_layer_call_fn_7903
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
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_78902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
Ä0
Æ
F__inference_functional_1_layer_call_and_return_conditional_losses_7863
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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢lambda/StatefulPartitionedCall¢0variational_log_variance/StatefulPartitionedCall¢(variational_mean/StatefulPartitionedCall 
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_76372 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_75852
max_pooling2d/PartitionedCallÍ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_76662"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_76022!
max_pooling2d_1/PartitionedCallÏ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_76952"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_76192!
max_pooling2d_2/PartitionedCallõ
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_77192
flatten/PartitionedCall÷
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_variational_mean_layer_call_and_return_conditional_losses_77372*
(variational_mean/StatefulPartitionedCall¯
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_775922
0variational_log_variance/StatefulPartitionedCallÎ
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_78232 
lambda/StatefulPartitionedCallá
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
¦
 
/__inference_variational_mean_layer_call_fn_9672

inputs
variational_mean_kernel
variational_mean_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsvariational_mean_kernelvariational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_variational_mean_layer_call_and_return_conditional_losses_77372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
Å
+__inference_functional_5_layer_call_fn_8747
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
variational_log_variance_bias
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_biasconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_87262
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
Û
ý
F__inference_functional_5_layer_call_and_return_conditional_losses_8679

inputs
functional_1_conv2d_kernel
functional_1_conv2d_bias 
functional_1_conv2d_1_kernel
functional_1_conv2d_1_bias 
functional_1_conv2d_2_kernel
functional_1_conv2d_2_bias(
$functional_1_variational_mean_kernel&
"functional_1_variational_mean_bias0
,functional_1_variational_log_variance_kernel.
*functional_1_variational_log_variance_bias(
$functional_3_conv2d_transpose_kernel&
"functional_3_conv2d_transpose_bias*
&functional_3_conv2d_transpose_1_kernel(
$functional_3_conv2d_transpose_1_bias*
&functional_3_conv2d_transpose_2_kernel(
$functional_3_conv2d_transpose_2_bias*
&functional_3_conv2d_transpose_3_kernel(
$functional_3_conv2d_transpose_3_bias
identity¢$functional_1/StatefulPartitionedCall¢$functional_3/StatefulPartitionedCallÝ
$functional_1/StatefulPartitionedCallStatefulPartitionedCallinputsfunctional_1_conv2d_kernelfunctional_1_conv2d_biasfunctional_1_conv2d_1_kernelfunctional_1_conv2d_1_biasfunctional_1_conv2d_2_kernelfunctional_1_conv2d_2_bias$functional_1_variational_mean_kernel"functional_1_variational_mean_bias,functional_1_variational_log_variance_kernel*functional_1_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_78902&
$functional_1/StatefulPartitionedCall
$functional_3/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0$functional_3_conv2d_transpose_kernel"functional_3_conv2d_transpose_bias&functional_3_conv2d_transpose_1_kernel$functional_3_conv2d_transpose_1_bias&functional_3_conv2d_transpose_2_kernel$functional_3_conv2d_transpose_2_bias&functional_3_conv2d_transpose_3_kernel$functional_3_conv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85032&
$functional_3/StatefulPartitionedCallé
IdentityIdentity-functional_3/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall%^functional_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2L
$functional_3/StatefulPartitionedCall$functional_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs
¶
æ
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_9682

inputs9
5matmul_readvariableop_variational_log_variance_kernel8
4biasadd_readvariableop_variational_log_variance_bias
identity¥
MatMul/ReadVariableOpReadVariableOp5matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul¡
BiasAdd/ReadVariableOpReadVariableOp4biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
]
A__inference_reshape_layer_call_and_return_conditional_losses_9767

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
strided_slice/stack_2â
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
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


'__inference_conv2d_2_layer_call_fn_9644

inputs
conv2d_2_kernel
conv2d_2_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernelconv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_76952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ		@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs

o
@__inference_lambda_layer_call_and_return_conditional_losses_9715
inputs_0
inputs_1
identityF
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
strided_slice/stack_2â
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
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
 *  ?2
random_normal/stddevå
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ç³¬2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿd2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/1

e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8287

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó
Å
+__inference_functional_5_layer_call_fn_8700
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
variational_log_variance_bias
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasconv2d_2_kernelconv2d_2_biasvariational_mean_kernelvariational_mean_biasvariational_log_variance_kernelvariational_log_variance_biasconv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_86792
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
&
Þ
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8138

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel2
.biasadd_readvariableop_conv2d_transpose_1_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Ä
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ù%
Ø
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8022

inputs;
7conv2d_transpose_readvariableop_conv2d_transpose_kernel0
,biasadd_readvariableop_conv2d_transpose_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Â
conv2d_transpose/ReadVariableOpReadVariableOp7conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ç

Ñ
+__inference_functional_3_layer_call_fn_8514
input_2
conv2d_transpose_kernel
conv2d_transpose_bias
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
conv2d_transpose_2_kernel
conv2d_transpose_2_bias
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_kernelconv2d_transpose_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasconv2d_transpose_2_kernelconv2d_transpose_2_biasconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_2

¦
1__inference_conv2d_transpose_3_layer_call_fn_8375

inputs
conv2d_transpose_3_kernel
conv2d_transpose_3_bias
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_kernelconv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_83702
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä0
Æ
F__inference_functional_1_layer_call_and_return_conditional_losses_7839
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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢lambda/StatefulPartitionedCall¢0variational_log_variance/StatefulPartitionedCall¢(variational_mean/StatefulPartitionedCall 
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_76372 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_75852
max_pooling2d/PartitionedCallÍ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_76662"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_76022!
max_pooling2d_1/PartitionedCallÏ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_76952"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_76192!
max_pooling2d_2/PartitionedCallõ
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_77192
flatten/PartitionedCall÷
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_variational_mean_layer_call_and_return_conditional_losses_77372*
(variational_mean/StatefulPartitionedCall¯
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_775922
0variational_log_variance/StatefulPartitionedCallÎ
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_77972 
lambda/StatefulPartitionedCallá
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
&
Þ
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_8254

inputs=
9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel2
.biasadd_readvariableop_conv2d_transpose_2_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Ä
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
*

F__inference_functional_3_layer_call_and_return_conditional_losses_8480
input_2,
(conv2d_transpose_conv2d_transpose_kernel*
&conv2d_transpose_conv2d_transpose_bias0
,conv2d_transpose_1_conv2d_transpose_1_kernel.
*conv2d_transpose_1_conv2d_transpose_1_bias0
,conv2d_transpose_2_conv2d_transpose_2_kernel.
*conv2d_transpose_2_conv2d_transpose_2_bias0
,conv2d_transpose_3_conv2d_transpose_3_kernel.
*conv2d_transpose_3_conv2d_transpose_3_bias
identity¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCallÛ
reshape/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_83932
reshape/PartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_80222*
(conv2d_transpose/StatefulPartitionedCall©
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_80552
up_sampling2d/PartitionedCall¥
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_81382,
*conv2d_transpose_1/StatefulPartitionedCall±
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_81712!
up_sampling2d_1/PartitionedCall§
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0,conv2d_transpose_2_conv2d_transpose_2_kernel*conv2d_transpose_2_conv2d_transpose_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_82542,
*conv2d_transpose_2/StatefulPartitionedCall±
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_82872!
up_sampling2d_2/PartitionedCall§
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0,conv2d_transpose_3_conv2d_transpose_3_kernel*conv2d_transpose_3_conv2d_transpose_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_83702,
*conv2d_transpose_3/StatefulPartitionedCallÓ
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿd::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_2

c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8040

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
J
.__inference_max_pooling2d_1_layer_call_fn_7605

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_76022
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
J__inference_variational_mean_layer_call_and_return_conditional_losses_9665

inputs1
-matmul_readvariableop_variational_mean_kernel0
,biasadd_readvariableop_variational_mean_bias
identity
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7585

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù%
Ø
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7981

inputs;
7conv2d_transpose_readvariableop_conv2d_transpose_kernel0
,biasadd_readvariableop_conv2d_transpose_bias
identityD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stack/3
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Â
conv2d_transpose/ReadVariableOpReadVariableOp7conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
â
]
A__inference_reshape_layer_call_and_return_conditional_losses_8393

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
strided_slice/stack_2â
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
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¨
H
,__inference_max_pooling2d_layer_call_fn_7588

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_75852
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

n
%__inference_lambda_layer_call_fn_9753
inputs_0
inputs_1
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_78232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/1

c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8055

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
J
.__inference_max_pooling2d_2_layer_call_fn_7622

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_76192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½Ö
Ê
__inference__wrapped_model_7571
input_1H
Dfunctional_5_functional_1_conv2d_conv2d_readvariableop_conv2d_kernelG
Cfunctional_5_functional_1_conv2d_biasadd_readvariableop_conv2d_biasL
Hfunctional_5_functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernelK
Gfunctional_5_functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_biasL
Hfunctional_5_functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernelK
Gfunctional_5_functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_bias\
Xfunctional_5_functional_1_variational_mean_matmul_readvariableop_variational_mean_kernel[
Wfunctional_5_functional_1_variational_mean_biasadd_readvariableop_variational_mean_biasl
hfunctional_5_functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernelk
gfunctional_5_functional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_biasf
bfunctional_5_functional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel[
Wfunctional_5_functional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_biasj
ffunctional_5_functional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel_
[functional_5_functional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_biasj
ffunctional_5_functional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel_
[functional_5_functional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_biasj
ffunctional_5_functional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel_
[functional_5_functional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias
identityý
6functional_5/functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOpDfunctional_5_functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:@*
dtype028
6functional_5/functional_1/conv2d/Conv2D/ReadVariableOp
'functional_5/functional_1/conv2d/Conv2DConv2Dinput_1>functional_5/functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
2)
'functional_5/functional_1/conv2d/Conv2Dò
7functional_5/functional_1/conv2d/BiasAdd/ReadVariableOpReadVariableOpCfunctional_5_functional_1_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:@*
dtype029
7functional_5/functional_1/conv2d/BiasAdd/ReadVariableOp
(functional_5/functional_1/conv2d/BiasAddBiasAdd0functional_5/functional_1/conv2d/Conv2D:output:0?functional_5/functional_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2*
(functional_5/functional_1/conv2d/BiasAddÃ
%functional_5/functional_1/conv2d/ReluRelu1functional_5/functional_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@2'
%functional_5/functional_1/conv2d/Relu
/functional_5/functional_1/max_pooling2d/MaxPoolMaxPool3functional_5/functional_1/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
21
/functional_5/functional_1/max_pooling2d/MaxPool
8functional_5/functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpHfunctional_5_functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02:
8functional_5/functional_1/conv2d_1/Conv2D/ReadVariableOp¿
)functional_5/functional_1/conv2d_1/Conv2DConv2D8functional_5/functional_1/max_pooling2d/MaxPool:output:0@functional_5/functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2+
)functional_5/functional_1/conv2d_1/Conv2Dú
9functional_5/functional_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGfunctional_5_functional_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:@*
dtype02;
9functional_5/functional_1/conv2d_1/BiasAdd/ReadVariableOp
*functional_5/functional_1/conv2d_1/BiasAddBiasAdd2functional_5/functional_1/conv2d_1/Conv2D:output:0Afunctional_5/functional_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_5/functional_1/conv2d_1/BiasAddÉ
'functional_5/functional_1/conv2d_1/ReluRelu3functional_5/functional_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'functional_5/functional_1/conv2d_1/Relu
1functional_5/functional_1/max_pooling2d_1/MaxPoolMaxPool5functional_5/functional_1/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
23
1functional_5/functional_1/max_pooling2d_1/MaxPool
8functional_5/functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpHfunctional_5_functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:@ *
dtype02:
8functional_5/functional_1/conv2d_2/Conv2D/ReadVariableOpÁ
)functional_5/functional_1/conv2d_2/Conv2DConv2D:functional_5/functional_1/max_pooling2d_1/MaxPool:output:0@functional_5/functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2+
)functional_5/functional_1/conv2d_2/Conv2Dú
9functional_5/functional_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGfunctional_5_functional_1_conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
: *
dtype02;
9functional_5/functional_1/conv2d_2/BiasAdd/ReadVariableOp
*functional_5/functional_1/conv2d_2/BiasAddBiasAdd2functional_5/functional_1/conv2d_2/Conv2D:output:0Afunctional_5/functional_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_5/functional_1/conv2d_2/BiasAddÉ
'functional_5/functional_1/conv2d_2/ReluRelu3functional_5/functional_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'functional_5/functional_1/conv2d_2/Relu
1functional_5/functional_1/max_pooling2d_2/MaxPoolMaxPool5functional_5/functional_1/conv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
23
1functional_5/functional_1/max_pooling2d_2/MaxPool£
'functional_5/functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2)
'functional_5/functional_1/flatten/Const
)functional_5/functional_1/flatten/ReshapeReshape:functional_5/functional_1/max_pooling2d_2/MaxPool:output:00functional_5/functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)functional_5/functional_1/flatten/Reshape
@functional_5/functional_1/variational_mean/MatMul/ReadVariableOpReadVariableOpXfunctional_5_functional_1_variational_mean_matmul_readvariableop_variational_mean_kernel*
_output_shapes
:	 d*
dtype02B
@functional_5/functional_1/variational_mean/MatMul/ReadVariableOp 
1functional_5/functional_1/variational_mean/MatMulMatMul2functional_5/functional_1/flatten/Reshape:output:0Hfunctional_5/functional_1/variational_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd23
1functional_5/functional_1/variational_mean/MatMul
Afunctional_5/functional_1/variational_mean/BiasAdd/ReadVariableOpReadVariableOpWfunctional_5_functional_1_variational_mean_biasadd_readvariableop_variational_mean_bias*
_output_shapes
:d*
dtype02C
Afunctional_5/functional_1/variational_mean/BiasAdd/ReadVariableOp­
2functional_5/functional_1/variational_mean/BiasAddBiasAdd;functional_5/functional_1/variational_mean/MatMul:product:0Ifunctional_5/functional_1/variational_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd24
2functional_5/functional_1/variational_mean/BiasAdd¾
Hfunctional_5/functional_1/variational_log_variance/MatMul/ReadVariableOpReadVariableOphfunctional_5_functional_1_variational_log_variance_matmul_readvariableop_variational_log_variance_kernel*
_output_shapes
:	 d*
dtype02J
Hfunctional_5/functional_1/variational_log_variance/MatMul/ReadVariableOp¸
9functional_5/functional_1/variational_log_variance/MatMulMatMul2functional_5/functional_1/flatten/Reshape:output:0Pfunctional_5/functional_1/variational_log_variance/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2;
9functional_5/functional_1/variational_log_variance/MatMulº
Ifunctional_5/functional_1/variational_log_variance/BiasAdd/ReadVariableOpReadVariableOpgfunctional_5_functional_1_variational_log_variance_biasadd_readvariableop_variational_log_variance_bias*
_output_shapes
:d*
dtype02K
Ifunctional_5/functional_1/variational_log_variance/BiasAdd/ReadVariableOpÍ
:functional_5/functional_1/variational_log_variance/BiasAddBiasAddCfunctional_5/functional_1/variational_log_variance/MatMul:product:0Qfunctional_5/functional_1/variational_log_variance/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2<
:functional_5/functional_1/variational_log_variance/BiasAddÃ
&functional_5/functional_1/lambda/ShapeShapeCfunctional_5/functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2(
&functional_5/functional_1/lambda/Shape¶
4functional_5/functional_1/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4functional_5/functional_1/lambda/strided_slice/stackº
6functional_5/functional_1/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_5/functional_1/lambda/strided_slice/stack_1º
6functional_5/functional_1/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_5/functional_1/lambda/strided_slice/stack_2¨
.functional_5/functional_1/lambda/strided_sliceStridedSlice/functional_5/functional_1/lambda/Shape:output:0=functional_5/functional_1/lambda/strided_slice/stack:output:0?functional_5/functional_1/lambda/strided_slice/stack_1:output:0?functional_5/functional_1/lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.functional_5/functional_1/lambda/strided_sliceÇ
(functional_5/functional_1/lambda/Shape_1ShapeCfunctional_5/functional_1/variational_log_variance/BiasAdd:output:0*
T0*
_output_shapes
:2*
(functional_5/functional_1/lambda/Shape_1º
6functional_5/functional_1/lambda/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6functional_5/functional_1/lambda/strided_slice_1/stack¾
8functional_5/functional_1/lambda/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8functional_5/functional_1/lambda/strided_slice_1/stack_1¾
8functional_5/functional_1/lambda/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8functional_5/functional_1/lambda/strided_slice_1/stack_2´
0functional_5/functional_1/lambda/strided_slice_1StridedSlice1functional_5/functional_1/lambda/Shape_1:output:0?functional_5/functional_1/lambda/strided_slice_1/stack:output:0Afunctional_5/functional_1/lambda/strided_slice_1/stack_1:output:0Afunctional_5/functional_1/lambda/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0functional_5/functional_1/lambda/strided_slice_1
4functional_5/functional_1/lambda/random_normal/shapePack7functional_5/functional_1/lambda/strided_slice:output:09functional_5/functional_1/lambda/strided_slice_1:output:0*
N*
T0*
_output_shapes
:26
4functional_5/functional_1/lambda/random_normal/shape¯
3functional_5/functional_1/lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3functional_5/functional_1/lambda/random_normal/mean³
5functional_5/functional_1/lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5functional_5/functional_1/lambda/random_normal/stddevÈ
Cfunctional_5/functional_1/lambda/random_normal/RandomStandardNormalRandomStandardNormal=functional_5/functional_1/lambda/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ð2E
Cfunctional_5/functional_1/lambda/random_normal/RandomStandardNormal¸
2functional_5/functional_1/lambda/random_normal/mulMulLfunctional_5/functional_1/lambda/random_normal/RandomStandardNormal:output:0>functional_5/functional_1/lambda/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ24
2functional_5/functional_1/lambda/random_normal/mul
.functional_5/functional_1/lambda/random_normalAdd6functional_5/functional_1/lambda/random_normal/mul:z:0<functional_5/functional_1/lambda/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ20
.functional_5/functional_1/lambda/random_normal
&functional_5/functional_1/lambda/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&functional_5/functional_1/lambda/mul/xû
$functional_5/functional_1/lambda/mulMul/functional_5/functional_1/lambda/mul/x:output:0Cfunctional_5/functional_1/variational_log_variance/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$functional_5/functional_1/lambda/mul¯
$functional_5/functional_1/lambda/ExpExp(functional_5/functional_1/lambda/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$functional_5/functional_1/lambda/Expç
&functional_5/functional_1/lambda/mul_1Mul(functional_5/functional_1/lambda/Exp:y:02functional_5/functional_1/lambda/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2(
&functional_5/functional_1/lambda/mul_1ð
$functional_5/functional_1/lambda/addAddV2;functional_5/functional_1/variational_mean/BiasAdd:output:0*functional_5/functional_1/lambda/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$functional_5/functional_1/lambda/addª
'functional_5/functional_3/reshape/ShapeShape(functional_5/functional_1/lambda/add:z:0*
T0*
_output_shapes
:2)
'functional_5/functional_3/reshape/Shape¸
5functional_5/functional_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5functional_5/functional_3/reshape/strided_slice/stack¼
7functional_5/functional_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_5/functional_3/reshape/strided_slice/stack_1¼
7functional_5/functional_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7functional_5/functional_3/reshape/strided_slice/stack_2®
/functional_5/functional_3/reshape/strided_sliceStridedSlice0functional_5/functional_3/reshape/Shape:output:0>functional_5/functional_3/reshape/strided_slice/stack:output:0@functional_5/functional_3/reshape/strided_slice/stack_1:output:0@functional_5/functional_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/functional_5/functional_3/reshape/strided_slice¨
1functional_5/functional_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1functional_5/functional_3/reshape/Reshape/shape/1¨
1functional_5/functional_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1functional_5/functional_3/reshape/Reshape/shape/2¨
1functional_5/functional_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :d23
1functional_5/functional_3/reshape/Reshape/shape/3
/functional_5/functional_3/reshape/Reshape/shapePack8functional_5/functional_3/reshape/strided_slice:output:0:functional_5/functional_3/reshape/Reshape/shape/1:output:0:functional_5/functional_3/reshape/Reshape/shape/2:output:0:functional_5/functional_3/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/functional_5/functional_3/reshape/Reshape/shapeÿ
)functional_5/functional_3/reshape/ReshapeReshape(functional_5/functional_1/lambda/add:z:08functional_5/functional_3/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2+
)functional_5/functional_3/reshape/ReshapeÆ
0functional_5/functional_3/conv2d_transpose/ShapeShape2functional_5/functional_3/reshape/Reshape:output:0*
T0*
_output_shapes
:22
0functional_5/functional_3/conv2d_transpose/ShapeÊ
>functional_5/functional_3/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>functional_5/functional_3/conv2d_transpose/strided_slice/stackÎ
@functional_5/functional_3/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@functional_5/functional_3/conv2d_transpose/strided_slice/stack_1Î
@functional_5/functional_3/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@functional_5/functional_3/conv2d_transpose/strided_slice/stack_2ä
8functional_5/functional_3/conv2d_transpose/strided_sliceStridedSlice9functional_5/functional_3/conv2d_transpose/Shape:output:0Gfunctional_5/functional_3/conv2d_transpose/strided_slice/stack:output:0Ifunctional_5/functional_3/conv2d_transpose/strided_slice/stack_1:output:0Ifunctional_5/functional_3/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8functional_5/functional_3/conv2d_transpose/strided_sliceª
2functional_5/functional_3/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :24
2functional_5/functional_3/conv2d_transpose/stack/1ª
2functional_5/functional_3/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :24
2functional_5/functional_3/conv2d_transpose/stack/2ª
2functional_5/functional_3/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@24
2functional_5/functional_3/conv2d_transpose/stack/3
0functional_5/functional_3/conv2d_transpose/stackPackAfunctional_5/functional_3/conv2d_transpose/strided_slice:output:0;functional_5/functional_3/conv2d_transpose/stack/1:output:0;functional_5/functional_3/conv2d_transpose/stack/2:output:0;functional_5/functional_3/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:22
0functional_5/functional_3/conv2d_transpose/stackÎ
@functional_5/functional_3/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@functional_5/functional_3/conv2d_transpose/strided_slice_1/stackÒ
Bfunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack_1Ò
Bfunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack_2î
:functional_5/functional_3/conv2d_transpose/strided_slice_1StridedSlice9functional_5/functional_3/conv2d_transpose/stack:output:0Ifunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack:output:0Kfunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack_1:output:0Kfunctional_5/functional_3/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:functional_5/functional_3/conv2d_transpose/strided_slice_1Ã
Jfunctional_5/functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpbfunctional_5_functional_3_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@d*
dtype02L
Jfunctional_5/functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOp·
;functional_5/functional_3/conv2d_transpose/conv2d_transposeConv2DBackpropInput9functional_5/functional_3/conv2d_transpose/stack:output:0Rfunctional_5/functional_3/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:02functional_5/functional_3/reshape/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2=
;functional_5/functional_3/conv2d_transpose/conv2d_transpose
Afunctional_5/functional_3/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpWfunctional_5_functional_3_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype02C
Afunctional_5/functional_3/conv2d_transpose/BiasAdd/ReadVariableOp¾
2functional_5/functional_3/conv2d_transpose/BiasAddBiasAddDfunctional_5/functional_3/conv2d_transpose/conv2d_transpose:output:0Ifunctional_5/functional_3/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2functional_5/functional_3/conv2d_transpose/BiasAddá
/functional_5/functional_3/conv2d_transpose/ReluRelu;functional_5/functional_3/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@21
/functional_5/functional_3/conv2d_transpose/ReluË
-functional_5/functional_3/up_sampling2d/ShapeShape=functional_5/functional_3/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2/
-functional_5/functional_3/up_sampling2d/ShapeÄ
;functional_5/functional_3/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;functional_5/functional_3/up_sampling2d/strided_slice/stackÈ
=functional_5/functional_3/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=functional_5/functional_3/up_sampling2d/strided_slice/stack_1È
=functional_5/functional_3/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=functional_5/functional_3/up_sampling2d/strided_slice/stack_2¾
5functional_5/functional_3/up_sampling2d/strided_sliceStridedSlice6functional_5/functional_3/up_sampling2d/Shape:output:0Dfunctional_5/functional_3/up_sampling2d/strided_slice/stack:output:0Ffunctional_5/functional_3/up_sampling2d/strided_slice/stack_1:output:0Ffunctional_5/functional_3/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:27
5functional_5/functional_3/up_sampling2d/strided_slice¯
-functional_5/functional_3/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2/
-functional_5/functional_3/up_sampling2d/Constþ
+functional_5/functional_3/up_sampling2d/mulMul>functional_5/functional_3/up_sampling2d/strided_slice:output:06functional_5/functional_3/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2-
+functional_5/functional_3/up_sampling2d/mulé
Dfunctional_5/functional_3/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor=functional_5/functional_3/conv2d_transpose/Relu:activations:0/functional_5/functional_3/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2F
Dfunctional_5/functional_3/up_sampling2d/resize/ResizeNearestNeighborí
2functional_5/functional_3/conv2d_transpose_1/ShapeShapeUfunctional_5/functional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:24
2functional_5/functional_3/conv2d_transpose_1/ShapeÎ
@functional_5/functional_3/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@functional_5/functional_3/conv2d_transpose_1/strided_slice/stackÒ
Bfunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack_1Ò
Bfunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack_2ð
:functional_5/functional_3/conv2d_transpose_1/strided_sliceStridedSlice;functional_5/functional_3/conv2d_transpose_1/Shape:output:0Ifunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack:output:0Kfunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack_1:output:0Kfunctional_5/functional_3/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:functional_5/functional_3/conv2d_transpose_1/strided_slice®
4functional_5/functional_3/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_5/functional_3/conv2d_transpose_1/stack/1®
4functional_5/functional_3/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_5/functional_3/conv2d_transpose_1/stack/2®
4functional_5/functional_3/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 26
4functional_5/functional_3/conv2d_transpose_1/stack/3 
2functional_5/functional_3/conv2d_transpose_1/stackPackCfunctional_5/functional_3/conv2d_transpose_1/strided_slice:output:0=functional_5/functional_3/conv2d_transpose_1/stack/1:output:0=functional_5/functional_3/conv2d_transpose_1/stack/2:output:0=functional_5/functional_3/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:24
2functional_5/functional_3/conv2d_transpose_1/stackÒ
Bfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stackÖ
Dfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack_1Ö
Dfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack_2ú
<functional_5/functional_3/conv2d_transpose_1/strided_slice_1StridedSlice;functional_5/functional_3/conv2d_transpose_1/stack:output:0Kfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack:output:0Mfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack_1:output:0Mfunctional_5/functional_3/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<functional_5/functional_3/conv2d_transpose_1/strided_slice_1Ë
Lfunctional_5/functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpffunctional_5_functional_3_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype02N
Lfunctional_5/functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOpâ
=functional_5/functional_3/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput;functional_5/functional_3/conv2d_transpose_1/stack:output:0Tfunctional_5/functional_3/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Ufunctional_5/functional_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2?
=functional_5/functional_3/conv2d_transpose_1/conv2d_transpose¢
Cfunctional_5/functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp[functional_5_functional_3_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype02E
Cfunctional_5/functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOpÆ
4functional_5/functional_3/conv2d_transpose_1/BiasAddBiasAddFfunctional_5/functional_3/conv2d_transpose_1/conv2d_transpose:output:0Kfunctional_5/functional_3/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4functional_5/functional_3/conv2d_transpose_1/BiasAddç
1functional_5/functional_3/conv2d_transpose_1/ReluRelu=functional_5/functional_3/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1functional_5/functional_3/conv2d_transpose_1/ReluÑ
/functional_5/functional_3/up_sampling2d_1/ShapeShape?functional_5/functional_3/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:21
/functional_5/functional_3/up_sampling2d_1/ShapeÈ
=functional_5/functional_3/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=functional_5/functional_3/up_sampling2d_1/strided_slice/stackÌ
?functional_5/functional_3/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?functional_5/functional_3/up_sampling2d_1/strided_slice/stack_1Ì
?functional_5/functional_3/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?functional_5/functional_3/up_sampling2d_1/strided_slice/stack_2Ê
7functional_5/functional_3/up_sampling2d_1/strided_sliceStridedSlice8functional_5/functional_3/up_sampling2d_1/Shape:output:0Ffunctional_5/functional_3/up_sampling2d_1/strided_slice/stack:output:0Hfunctional_5/functional_3/up_sampling2d_1/strided_slice/stack_1:output:0Hfunctional_5/functional_3/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:29
7functional_5/functional_3/up_sampling2d_1/strided_slice³
/functional_5/functional_3/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      21
/functional_5/functional_3/up_sampling2d_1/Const
-functional_5/functional_3/up_sampling2d_1/mulMul@functional_5/functional_3/up_sampling2d_1/strided_slice:output:08functional_5/functional_3/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2/
-functional_5/functional_3/up_sampling2d_1/mulñ
Ffunctional_5/functional_3/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor?functional_5/functional_3/conv2d_transpose_1/Relu:activations:01functional_5/functional_3/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2H
Ffunctional_5/functional_3/up_sampling2d_1/resize/ResizeNearestNeighborï
2functional_5/functional_3/conv2d_transpose_2/ShapeShapeWfunctional_5/functional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:24
2functional_5/functional_3/conv2d_transpose_2/ShapeÎ
@functional_5/functional_3/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@functional_5/functional_3/conv2d_transpose_2/strided_slice/stackÒ
Bfunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack_1Ò
Bfunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack_2ð
:functional_5/functional_3/conv2d_transpose_2/strided_sliceStridedSlice;functional_5/functional_3/conv2d_transpose_2/Shape:output:0Ifunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack:output:0Kfunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack_1:output:0Kfunctional_5/functional_3/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:functional_5/functional_3/conv2d_transpose_2/strided_slice®
4functional_5/functional_3/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_5/functional_3/conv2d_transpose_2/stack/1®
4functional_5/functional_3/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_5/functional_3/conv2d_transpose_2/stack/2®
4functional_5/functional_3/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_5/functional_3/conv2d_transpose_2/stack/3 
2functional_5/functional_3/conv2d_transpose_2/stackPackCfunctional_5/functional_3/conv2d_transpose_2/strided_slice:output:0=functional_5/functional_3/conv2d_transpose_2/stack/1:output:0=functional_5/functional_3/conv2d_transpose_2/stack/2:output:0=functional_5/functional_3/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:24
2functional_5/functional_3/conv2d_transpose_2/stackÒ
Bfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stackÖ
Dfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack_1Ö
Dfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack_2ú
<functional_5/functional_3/conv2d_transpose_2/strided_slice_1StridedSlice;functional_5/functional_3/conv2d_transpose_2/stack:output:0Kfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack:output:0Mfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack_1:output:0Mfunctional_5/functional_3/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<functional_5/functional_3/conv2d_transpose_2/strided_slice_1Ë
Lfunctional_5/functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpffunctional_5_functional_3_conv2d_transpose_2_conv2d_transpose_readvariableop_conv2d_transpose_2_kernel*&
_output_shapes
: *
dtype02N
Lfunctional_5/functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOpä
=functional_5/functional_3/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput;functional_5/functional_3/conv2d_transpose_2/stack:output:0Tfunctional_5/functional_3/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0Wfunctional_5/functional_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2?
=functional_5/functional_3/conv2d_transpose_2/conv2d_transpose¢
Cfunctional_5/functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp[functional_5_functional_3_conv2d_transpose_2_biasadd_readvariableop_conv2d_transpose_2_bias*
_output_shapes
:*
dtype02E
Cfunctional_5/functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOpÆ
4functional_5/functional_3/conv2d_transpose_2/BiasAddBiasAddFfunctional_5/functional_3/conv2d_transpose_2/conv2d_transpose:output:0Kfunctional_5/functional_3/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4functional_5/functional_3/conv2d_transpose_2/BiasAddç
1functional_5/functional_3/conv2d_transpose_2/ReluRelu=functional_5/functional_3/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1functional_5/functional_3/conv2d_transpose_2/ReluÑ
/functional_5/functional_3/up_sampling2d_2/ShapeShape?functional_5/functional_3/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:21
/functional_5/functional_3/up_sampling2d_2/ShapeÈ
=functional_5/functional_3/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=functional_5/functional_3/up_sampling2d_2/strided_slice/stackÌ
?functional_5/functional_3/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?functional_5/functional_3/up_sampling2d_2/strided_slice/stack_1Ì
?functional_5/functional_3/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?functional_5/functional_3/up_sampling2d_2/strided_slice/stack_2Ê
7functional_5/functional_3/up_sampling2d_2/strided_sliceStridedSlice8functional_5/functional_3/up_sampling2d_2/Shape:output:0Ffunctional_5/functional_3/up_sampling2d_2/strided_slice/stack:output:0Hfunctional_5/functional_3/up_sampling2d_2/strided_slice/stack_1:output:0Hfunctional_5/functional_3/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:29
7functional_5/functional_3/up_sampling2d_2/strided_slice³
/functional_5/functional_3/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      21
/functional_5/functional_3/up_sampling2d_2/Const
-functional_5/functional_3/up_sampling2d_2/mulMul@functional_5/functional_3/up_sampling2d_2/strided_slice:output:08functional_5/functional_3/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2/
-functional_5/functional_3/up_sampling2d_2/mulñ
Ffunctional_5/functional_3/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor?functional_5/functional_3/conv2d_transpose_2/Relu:activations:01functional_5/functional_3/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((*
half_pixel_centers(2H
Ffunctional_5/functional_3/up_sampling2d_2/resize/ResizeNearestNeighborï
2functional_5/functional_3/conv2d_transpose_3/ShapeShapeWfunctional_5/functional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:24
2functional_5/functional_3/conv2d_transpose_3/ShapeÎ
@functional_5/functional_3/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@functional_5/functional_3/conv2d_transpose_3/strided_slice/stackÒ
Bfunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack_1Ò
Bfunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bfunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack_2ð
:functional_5/functional_3/conv2d_transpose_3/strided_sliceStridedSlice;functional_5/functional_3/conv2d_transpose_3/Shape:output:0Ifunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack:output:0Kfunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack_1:output:0Kfunctional_5/functional_3/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:functional_5/functional_3/conv2d_transpose_3/strided_slice®
4functional_5/functional_3/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :-26
4functional_5/functional_3/conv2d_transpose_3/stack/1®
4functional_5/functional_3/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :-26
4functional_5/functional_3/conv2d_transpose_3/stack/2®
4functional_5/functional_3/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_5/functional_3/conv2d_transpose_3/stack/3 
2functional_5/functional_3/conv2d_transpose_3/stackPackCfunctional_5/functional_3/conv2d_transpose_3/strided_slice:output:0=functional_5/functional_3/conv2d_transpose_3/stack/1:output:0=functional_5/functional_3/conv2d_transpose_3/stack/2:output:0=functional_5/functional_3/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:24
2functional_5/functional_3/conv2d_transpose_3/stackÒ
Bfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stackÖ
Dfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack_1Ö
Dfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack_2ú
<functional_5/functional_3/conv2d_transpose_3/strided_slice_1StridedSlice;functional_5/functional_3/conv2d_transpose_3/stack:output:0Kfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack:output:0Mfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack_1:output:0Mfunctional_5/functional_3/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<functional_5/functional_3/conv2d_transpose_3/strided_slice_1Ë
Lfunctional_5/functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpffunctional_5_functional_3_conv2d_transpose_3_conv2d_transpose_readvariableop_conv2d_transpose_3_kernel*&
_output_shapes
:*
dtype02N
Lfunctional_5/functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOpä
=functional_5/functional_3/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput;functional_5/functional_3/conv2d_transpose_3/stack:output:0Tfunctional_5/functional_3/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0Wfunctional_5/functional_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--*
paddingVALID*
strides
2?
=functional_5/functional_3/conv2d_transpose_3/conv2d_transpose¢
Cfunctional_5/functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp[functional_5_functional_3_conv2d_transpose_3_biasadd_readvariableop_conv2d_transpose_3_bias*
_output_shapes
:*
dtype02E
Cfunctional_5/functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOpÆ
4functional_5/functional_3/conv2d_transpose_3/BiasAddBiasAddFfunctional_5/functional_3/conv2d_transpose_3/conv2d_transpose:output:0Kfunctional_5/functional_3/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--26
4functional_5/functional_3/conv2d_transpose_3/BiasAddç
1functional_5/functional_3/conv2d_transpose_3/ReluRelu=functional_5/functional_3/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--23
1functional_5/functional_3/conv2d_transpose_3/Relu
IdentityIdentity?functional_5/functional_3/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--:::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
!
_user_specified_name	input_1
Û
ý
F__inference_functional_5_layer_call_and_return_conditional_losses_8726

inputs
functional_1_conv2d_kernel
functional_1_conv2d_bias 
functional_1_conv2d_1_kernel
functional_1_conv2d_1_bias 
functional_1_conv2d_2_kernel
functional_1_conv2d_2_bias(
$functional_1_variational_mean_kernel&
"functional_1_variational_mean_bias0
,functional_1_variational_log_variance_kernel.
*functional_1_variational_log_variance_bias(
$functional_3_conv2d_transpose_kernel&
"functional_3_conv2d_transpose_bias*
&functional_3_conv2d_transpose_1_kernel(
$functional_3_conv2d_transpose_1_bias*
&functional_3_conv2d_transpose_2_kernel(
$functional_3_conv2d_transpose_2_bias*
&functional_3_conv2d_transpose_3_kernel(
$functional_3_conv2d_transpose_3_bias
identity¢$functional_1/StatefulPartitionedCall¢$functional_3/StatefulPartitionedCallÝ
$functional_1/StatefulPartitionedCallStatefulPartitionedCallinputsfunctional_1_conv2d_kernelfunctional_1_conv2d_biasfunctional_1_conv2d_1_kernelfunctional_1_conv2d_1_biasfunctional_1_conv2d_2_kernelfunctional_1_conv2d_2_bias$functional_1_variational_mean_kernel"functional_1_variational_mean_bias,functional_1_variational_log_variance_kernel*functional_1_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_79292&
$functional_1/StatefulPartitionedCall
$functional_3/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0$functional_3_conv2d_transpose_kernel"functional_3_conv2d_transpose_bias&functional_3_conv2d_transpose_1_kernel$functional_3_conv2d_transpose_1_bias&functional_3_conv2d_transpose_2_kernel$functional_3_conv2d_transpose_2_bias&functional_3_conv2d_transpose_3_kernel$functional_3_conv2d_transpose_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_3_layer_call_and_return_conditional_losses_85362&
$functional_3/StatefulPartitionedCallé
IdentityIdentity-functional_3/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall%^functional_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ--::::::::::::::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2L
$functional_3/StatefulPartitionedCall$functional_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs

n
%__inference_lambda_layer_call_fn_9747
inputs_0
inputs_1
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_77972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/1
ý
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7577

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¸
7__inference_variational_log_variance_layer_call_fn_9689

inputs#
variational_log_variance_kernel!
variational_log_variance_bias
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsvariational_log_variance_kernelvariational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_77592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¡
B
&__inference_flatten_layer_call_fn_9655

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_77192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Á0
Å
F__inference_functional_1_layer_call_and_return_conditional_losses_7929

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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢lambda/StatefulPartitionedCall¢0variational_log_variance/StatefulPartitionedCall¢(variational_mean/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_76372 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_75852
max_pooling2d/PartitionedCallÍ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_76662"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_76022!
max_pooling2d_1/PartitionedCallÏ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_76952"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_76192!
max_pooling2d_2/PartitionedCallõ
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_77192
flatten/PartitionedCall÷
(variational_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0(variational_mean_variational_mean_kernel&variational_mean_variational_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_variational_mean_layer_call_and_return_conditional_losses_77372*
(variational_mean/StatefulPartitionedCall¯
0variational_log_variance/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:08variational_log_variance_variational_log_variance_kernel6variational_log_variance_variational_log_variance_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_775922
0variational_log_variance/StatefulPartitionedCallÎ
lambda/StatefulPartitionedCallStatefulPartitionedCall1variational_mean/StatefulPartitionedCall:output:09variational_log_variance/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_78232 
lambda/StatefulPartitionedCallá
IdentityIdentity'lambda/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall1^variational_log_variance/StatefulPartitionedCall)^variational_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ--::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2d
0variational_log_variance/StatefulPartitionedCall0variational_log_variance/StatefulPartitionedCall2T
(variational_mean/StatefulPartitionedCall(variational_mean/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--
 
_user_specified_nameinputs

¦
1__inference_conv2d_transpose_1_layer_call_fn_8143

inputs
conv2d_transpose_1_kernel
conv2d_transpose_1_bias
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_1_kernelconv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_81382
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
H
,__inference_up_sampling2d_layer_call_fn_8058

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_80552
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

 
/__inference_conv2d_transpose_layer_call_fn_8027

inputs
conv2d_transpose_kernel
conv2d_transpose_bias
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_80222
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ--H
functional_38
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ--tensorflow/serving/predict:³
´
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"á±
_tf_keras_networkÄ±{"class_name": "Functional", "name": "functional_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_log_variance", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["variational_mean", 0, 0, {}], ["variational_log_variance", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "name": "functional_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}, "name": "reshape", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}, "name": "functional_3", "inbound_nodes": [[["functional_1", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["functional_3", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 45, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_log_variance", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["variational_mean", 0, 0, {}], ["variational_log_variance", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "name": "functional_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}, "name": "reshape", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}, "name": "functional_3", "inbound_nodes": [[["functional_1", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["functional_3", 1, 0]]}}, "training_config": {"loss": "total_loss", "metrics": [], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ù"ö
_tf_keras_input_layerÖ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
©`
layer-0

layer_with_weights-0

layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
layer_with_weights-4
layer-9
layer-10
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"]
_tf_keras_networkê\{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_log_variance", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["variational_mean", 0, 0, {}], ["variational_log_variance", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 45, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 45, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "variational_log_variance", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["variational_mean", 0, 0, {}], ["variational_log_variance", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}}}
O
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
 layer_with_weights-3
 layer-8
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+&call_and_return_all_conditional_losses
__call__"ªL
_tf_keras_networkL{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}, "name": "reshape", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}, "name": "reshape", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}}}
»
%iter

&beta_1

'beta_2
	(decay
)learning_rate*mí+mî,mï-mð.mñ/mò0mó1mô2mõ3mö4m÷5mø6mù7mú8mû9mü:mý;mþ*vÿ+v,v-v.v/v0v1v2v3v4v5v6v7v8v9v:v;v"
	optimizer
 "
trackable_list_wrapper
¦
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17"
trackable_list_wrapper
¦
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17"
trackable_list_wrapper
Î
<layer_metrics
regularization_losses
=metrics

>layers
	variables
?non_trainable_variables
trainable_variables
@layer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
	

*kernel
+bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+&call_and_return_all_conditional_losses
__call__"ø
_tf_keras_layerÞ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}}
ý
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¤	

,kernel
-bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+&call_and_return_all_conditional_losses
__call__"ý
_tf_keras_layerã{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}

Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+&call_and_return_all_conditional_losses
 __call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¤	

.kernel
/bias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
+¡&call_and_return_all_conditional_losses
¢__call__"ý
_tf_keras_layerã{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}

Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+£&call_and_return_all_conditional_losses
¤__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ä
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+¥&call_and_return_all_conditional_losses
¦__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
¿

0kernel
1bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"
_tf_keras_layerþ{"class_name": "Dense", "name": "variational_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "variational_mean", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 288}}}}
Ï

2kernel
3bias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"¨
_tf_keras_layer{"class_name": "Dense", "name": "variational_log_variance", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "variational_log_variance", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 288}}}}
ß
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"Î
_tf_keras_layer´{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAABQAAAAQAAABDAAAAc0oAAAB8AFwCfQF9AnQAagF8AoMBZAEZAH0DdABqAmoDagR8\nA3QAagF8AoMBZAIZAGYCZAONAX0EfAF0AGoFZAR8AhQAgwF8BBQAFwBTACkFTukAAAAA6QEAAAAp\nAdoFc2hhcGVnAAAAAAAA4D8pBtoKdGVuc29yZmxvd3IDAAAA2gVrZXJhc9oHYmFja2VuZNoNcmFu\nZG9tX25vcm1hbNoDZXhwKQXaDGRpc3RyaWJ1dGlvbtoRZGlzdHJpYnV0aW9uX21lYW7aFWRpc3Ry\naWJ1dGlvbl92YXJpYW5jZdoKYmF0Y2hfc2l6ZdoGcmFuZG9tqQByDgAAAPofPGlweXRob24taW5w\ndXQtMjUtODg2M2MwY2UzODVjPtoWc2FtcGxlX2xhdGVudF9mZWF0dXJlcwEAAABzCAAAAAABCAEO\nAR4B\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
trackable_list_wrapper
f
*0
+1
,2
-3
.4
/5
06
17
28
39"
trackable_list_wrapper
f
*0
+1
,2
-3
.4
/5
06
17
28
39"
trackable_list_wrapper
°
ilayer_metrics
regularization_losses
jmetrics

klayers
	variables
lnon_trainable_variables
trainable_variables
mlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í"ê
_tf_keras_input_layerÊ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
÷
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+­&call_and_return_all_conditional_losses
®__call__"æ
_tf_keras_layerÌ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 100]}}}
Ö	

4kernel
5bias
rregularization_losses
strainable_variables
t	variables
u	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"¯
_tf_keras_layer{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 100}}}}
Ç
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
+±&call_and_return_all_conditional_losses
²__call__"¶
_tf_keras_layer{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ù	

6kernel
7bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
+³&call_and_return_all_conditional_losses
´__call__"²
_tf_keras_layer{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
Í
~regularization_losses
trainable_variables
	variables
	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"º
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ý	

8kernel
9bias
regularization_losses
trainable_variables
	variables
	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"²
_tf_keras_layer{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Ï
regularization_losses
trainable_variables
	variables
	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"º
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ü	

:kernel
;bias
regularization_losses
trainable_variables
	variables
	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"±
_tf_keras_layer{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
 "
trackable_list_wrapper
X
40
51
62
73
84
95
:6
;7"
trackable_list_wrapper
X
40
51
62
73
84
95
:6
;7"
trackable_list_wrapper
µ
layer_metrics
!regularization_losses
metrics
layers
"	variables
non_trainable_variables
#trainable_variables
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_6/Adam/iter
 : (2training_6/Adam/beta_1
 : (2training_6/Adam/beta_2
: (2training_6/Adam/decay
':% (2training_6/Adam/learning_rate
':%@2conv2d/kernel
:@2conv2d/bias
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
):'@ 2conv2d_2/kernel
: 2conv2d_2/bias
*:(	 d2variational_mean/kernel
#:!d2variational_mean/bias
2:0	 d2variational_log_variance/kernel
+:)d2variational_log_variance/bias
1:/@d2conv2d_transpose/kernel
#:!@2conv2d_transpose/bias
3:1 @2conv2d_transpose_1/kernel
%:# 2conv2d_transpose_1/bias
3:1 2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
3:12conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
µ
layer_metrics
Aregularization_losses
layers
Btrainable_variables
C	variables
non_trainable_variables
metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
Eregularization_losses
layers
Ftrainable_variables
G	variables
non_trainable_variables
metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
µ
layer_metrics
Iregularization_losses
layers
Jtrainable_variables
K	variables
non_trainable_variables
 metrics
 ¡layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¢layer_metrics
Mregularization_losses
£layers
Ntrainable_variables
O	variables
¤non_trainable_variables
¥metrics
 ¦layer_regularization_losses
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
µ
§layer_metrics
Qregularization_losses
¨layers
Rtrainable_variables
S	variables
©non_trainable_variables
ªmetrics
 «layer_regularization_losses
¢__call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬layer_metrics
Uregularization_losses
­layers
Vtrainable_variables
W	variables
®non_trainable_variables
¯metrics
 °layer_regularization_losses
¤__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±layer_metrics
Yregularization_losses
²layers
Ztrainable_variables
[	variables
³non_trainable_variables
´metrics
 µlayer_regularization_losses
¦__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
µ
¶layer_metrics
]regularization_losses
·layers
^trainable_variables
_	variables
¸non_trainable_variables
¹metrics
 ºlayer_regularization_losses
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
µ
»layer_metrics
aregularization_losses
¼layers
btrainable_variables
c	variables
½non_trainable_variables
¾metrics
 ¿layer_regularization_losses
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Àlayer_metrics
eregularization_losses
Álayers
ftrainable_variables
g	variables
Ânon_trainable_variables
Ãmetrics
 Älayer_regularization_losses
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
n
0

1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ålayer_metrics
nregularization_losses
Ælayers
otrainable_variables
p	variables
Çnon_trainable_variables
Èmetrics
 Élayer_regularization_losses
®__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
µ
Êlayer_metrics
rregularization_losses
Ëlayers
strainable_variables
t	variables
Ìnon_trainable_variables
Ímetrics
 Îlayer_regularization_losses
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ïlayer_metrics
vregularization_losses
Ðlayers
wtrainable_variables
x	variables
Ñnon_trainable_variables
Òmetrics
 Ólayer_regularization_losses
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
µ
Ôlayer_metrics
zregularization_losses
Õlayers
{trainable_variables
|	variables
Önon_trainable_variables
×metrics
 Ølayer_regularization_losses
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¶
Ùlayer_metrics
~regularization_losses
Úlayers
trainable_variables
	variables
Ûnon_trainable_variables
Ümetrics
 Ýlayer_regularization_losses
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
¸
Þlayer_metrics
regularization_losses
ßlayers
trainable_variables
	variables
ànon_trainable_variables
ámetrics
 âlayer_regularization_losses
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãlayer_metrics
regularization_losses
älayers
trainable_variables
	variables
ånon_trainable_variables
æmetrics
 çlayer_regularization_losses
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
¸
èlayer_metrics
regularization_losses
élayers
trainable_variables
	variables
ênon_trainable_variables
ëmetrics
 ìlayer_regularization_losses
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
 8"
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
7:5@2training_6/Adam/conv2d/kernel/m
):'@2training_6/Adam/conv2d/bias/m
9:7@@2!training_6/Adam/conv2d_1/kernel/m
+:)@2training_6/Adam/conv2d_1/bias/m
9:7@ 2!training_6/Adam/conv2d_2/kernel/m
+:) 2training_6/Adam/conv2d_2/bias/m
::8	 d2)training_6/Adam/variational_mean/kernel/m
3:1d2'training_6/Adam/variational_mean/bias/m
B:@	 d21training_6/Adam/variational_log_variance/kernel/m
;:9d2/training_6/Adam/variational_log_variance/bias/m
A:?@d2)training_6/Adam/conv2d_transpose/kernel/m
3:1@2'training_6/Adam/conv2d_transpose/bias/m
C:A @2+training_6/Adam/conv2d_transpose_1/kernel/m
5:3 2)training_6/Adam/conv2d_transpose_1/bias/m
C:A 2+training_6/Adam/conv2d_transpose_2/kernel/m
5:32)training_6/Adam/conv2d_transpose_2/bias/m
C:A2+training_6/Adam/conv2d_transpose_3/kernel/m
5:32)training_6/Adam/conv2d_transpose_3/bias/m
7:5@2training_6/Adam/conv2d/kernel/v
):'@2training_6/Adam/conv2d/bias/v
9:7@@2!training_6/Adam/conv2d_1/kernel/v
+:)@2training_6/Adam/conv2d_1/bias/v
9:7@ 2!training_6/Adam/conv2d_2/kernel/v
+:) 2training_6/Adam/conv2d_2/bias/v
::8	 d2)training_6/Adam/variational_mean/kernel/v
3:1d2'training_6/Adam/variational_mean/bias/v
B:@	 d21training_6/Adam/variational_log_variance/kernel/v
;:9d2/training_6/Adam/variational_log_variance/bias/v
A:?@d2)training_6/Adam/conv2d_transpose/kernel/v
3:1@2'training_6/Adam/conv2d_transpose/bias/v
C:A @2+training_6/Adam/conv2d_transpose_1/kernel/v
5:3 2)training_6/Adam/conv2d_transpose_1/bias/v
C:A 2+training_6/Adam/conv2d_transpose_2/kernel/v
5:32)training_6/Adam/conv2d_transpose_2/bias/v
C:A2+training_6/Adam/conv2d_transpose_3/kernel/v
5:32)training_6/Adam/conv2d_transpose_3/bias/v
æ2ã
F__inference_functional_5_layer_call_and_return_conditional_losses_9126
F__inference_functional_5_layer_call_and_return_conditional_losses_8652
F__inference_functional_5_layer_call_and_return_conditional_losses_8949
F__inference_functional_5_layer_call_and_return_conditional_losses_8628À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
å2â
__inference__wrapped_model_7571¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ--
ú2÷
+__inference_functional_5_layer_call_fn_8747
+__inference_functional_5_layer_call_fn_9172
+__inference_functional_5_layer_call_fn_9149
+__inference_functional_5_layer_call_fn_8700À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_functional_1_layer_call_and_return_conditional_losses_9298
F__inference_functional_1_layer_call_and_return_conditional_losses_9235
F__inference_functional_1_layer_call_and_return_conditional_losses_7863
F__inference_functional_1_layer_call_and_return_conditional_losses_7839À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_functional_1_layer_call_fn_9313
+__inference_functional_1_layer_call_fn_7942
+__inference_functional_1_layer_call_fn_9328
+__inference_functional_1_layer_call_fn_7903À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_functional_3_layer_call_and_return_conditional_losses_9564
F__inference_functional_3_layer_call_and_return_conditional_losses_8480
F__inference_functional_3_layer_call_and_return_conditional_losses_9446
F__inference_functional_3_layer_call_and_return_conditional_losses_8460À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_functional_3_layer_call_fn_9577
+__inference_functional_3_layer_call_fn_9590
+__inference_functional_3_layer_call_fn_8547
+__inference_functional_3_layer_call_fn_8514À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
1B/
"__inference_signature_wrapper_8772input_1
ê2ç
@__inference_conv2d_layer_call_and_return_conditional_losses_9601¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_conv2d_layer_call_fn_9608¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¯2¬
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7577à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
,__inference_max_pooling2d_layer_call_fn_7588à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì2é
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9619¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_conv2d_1_layer_call_fn_9626¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±2®
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7594à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_max_pooling2d_1_layer_call_fn_7605à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì2é
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9637¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_conv2d_2_layer_call_fn_9644¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±2®
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7611à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_max_pooling2d_2_layer_call_fn_7622à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ë2è
A__inference_flatten_layer_call_and_return_conditional_losses_9650¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_flatten_layer_call_fn_9655¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_variational_mean_layer_call_and_return_conditional_losses_9665¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_variational_mean_layer_call_fn_9672¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ü2ù
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_9682¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
á2Þ
7__inference_variational_log_variance_layer_call_fn_9689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ê2Ç
@__inference_lambda_layer_call_and_return_conditional_losses_9741
@__inference_lambda_layer_call_and_return_conditional_losses_9715À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
%__inference_lambda_layer_call_fn_9747
%__inference_lambda_layer_call_fn_9753À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_reshape_layer_call_and_return_conditional_losses_9767¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_reshape_layer_call_fn_9772¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©2¦
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7981×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
2
/__inference_conv2d_transpose_layer_call_fn_8027×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
¯2¬
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8040à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
,__inference_up_sampling2d_layer_call_fn_8058à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
«2¨
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8097×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
1__inference_conv2d_transpose_1_layer_call_fn_8143×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
±2®
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8156à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_1_layer_call_fn_8174à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
«2¨
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_8213×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
2
1__inference_conv2d_transpose_2_layer_call_fn_8259×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
±2®
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8272à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_up_sampling2d_2_layer_call_fn_8290à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
«2¨
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8329×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
1__inference_conv2d_transpose_3_layer_call_fn_8375×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
__inference__wrapped_model_7571*+,-./0123456789:;8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ--
ª "Cª@
>
functional_3.+
functional_3ÿÿÿÿÿÿÿÿÿ--²
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9619l,-7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_conv2d_1_layer_call_fn_9626_,-7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@²
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9637l./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
'__inference_conv2d_2_layer_call_fn_9644_./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª " ÿÿÿÿÿÿÿÿÿ °
@__inference_conv2d_layer_call_and_return_conditional_losses_9601l*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ--
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ))@
 
%__inference_conv2d_layer_call_fn_9608_*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ--
ª " ÿÿÿÿÿÿÿÿÿ))@á
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_809767I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¹
1__inference_conv2d_transpose_1_layer_call_fn_814367I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ á
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_821389I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
1__inference_conv2d_transpose_2_layer_call_fn_825989I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8329:;I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
1__inference_conv2d_transpose_3_layer_call_fn_8375:;I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_798145I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ·
/__inference_conv2d_transpose_layer_call_fn_802745I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
A__inference_flatten_layer_call_and_return_conditional_losses_9650a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 ~
&__inference_flatten_layer_call_fn_9655T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¿
F__inference_functional_1_layer_call_and_return_conditional_losses_7839u
*+,-./0123@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¿
F__inference_functional_1_layer_call_and_return_conditional_losses_7863u
*+,-./0123@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¾
F__inference_functional_1_layer_call_and_return_conditional_losses_9235t
*+,-./0123?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¾
F__inference_functional_1_layer_call_and_return_conditional_losses_9298t
*+,-./0123?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
+__inference_functional_1_layer_call_fn_7903h
*+,-./0123@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p

 
ª "ÿÿÿÿÿÿÿÿÿd
+__inference_functional_1_layer_call_fn_7942h
*+,-./0123@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
+__inference_functional_1_layer_call_fn_9313g
*+,-./0123?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p

 
ª "ÿÿÿÿÿÿÿÿÿd
+__inference_functional_1_layer_call_fn_9328g
*+,-./0123?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p 

 
ª "ÿÿÿÿÿÿÿÿÿdÐ
F__inference_functional_3_layer_call_and_return_conditional_losses_8460456789:;8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿd
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
F__inference_functional_3_layer_call_and_return_conditional_losses_8480456789:;8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿd
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¼
F__inference_functional_3_layer_call_and_return_conditional_losses_9446r456789:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ--
 ¼
F__inference_functional_3_layer_call_and_return_conditional_losses_9564r456789:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ--
 §
+__inference_functional_3_layer_call_fn_8514x456789:;8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿd
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
+__inference_functional_3_layer_call_fn_8547x456789:;8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿd
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
+__inference_functional_3_layer_call_fn_9577w456789:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
+__inference_functional_3_layer_call_fn_9590w456789:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
F__inference_functional_5_layer_call_and_return_conditional_losses_8628*+,-./0123456789:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 â
F__inference_functional_5_layer_call_and_return_conditional_losses_8652*+,-./0123456789:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ï
F__inference_functional_5_layer_call_and_return_conditional_losses_8949*+,-./0123456789:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ--
 Ï
F__inference_functional_5_layer_call_and_return_conditional_losses_9126*+,-./0123456789:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ--
 º
+__inference_functional_5_layer_call_fn_8700*+,-./0123456789:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
+__inference_functional_5_layer_call_fn_8747*+,-./0123456789:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ--
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
+__inference_functional_5_layer_call_fn_9149*+,-./0123456789:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
+__inference_functional_5_layer_call_fn_9172*+,-./0123456789:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ--
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
@__inference_lambda_layer_call_and_return_conditional_losses_9715b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿd
"
inputs/1ÿÿÿÿÿÿÿÿÿd

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Ð
@__inference_lambda_layer_call_and_return_conditional_losses_9741b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿd
"
inputs/1ÿÿÿÿÿÿÿÿÿd

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 §
%__inference_lambda_layer_call_fn_9747~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿd
"
inputs/1ÿÿÿÿÿÿÿÿÿd

 
p
ª "ÿÿÿÿÿÿÿÿÿd§
%__inference_lambda_layer_call_fn_9753~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿd
"
inputs/1ÿÿÿÿÿÿÿÿÿd

 
p 
ª "ÿÿÿÿÿÿÿÿÿdì
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7594R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_1_layer_call_fn_7605R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7611R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_2_layer_call_fn_7622R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7577R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_max_pooling2d_layer_call_fn_7588R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
A__inference_reshape_layer_call_and_return_conditional_losses_9767`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿd
 }
&__inference_reshape_layer_call_fn_9772S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª " ÿÿÿÿÿÿÿÿÿdÅ
"__inference_signature_wrapper_8772*+,-./0123456789:;C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ--"Cª@
>
functional_3.+
functional_3ÿÿÿÿÿÿÿÿÿ--ì
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8156R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_1_layer_call_fn_8174R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8272R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_up_sampling2d_2_layer_call_fn_8290R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8040R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_up_sampling2d_layer_call_fn_8058R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
R__inference_variational_log_variance_layer_call_and_return_conditional_losses_9682]230¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
7__inference_variational_log_variance_layer_call_fn_9689P230¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿd«
J__inference_variational_mean_layer_call_and_return_conditional_losses_9665]010¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
/__inference_variational_mean_layer_call_fn_9672P010¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿd