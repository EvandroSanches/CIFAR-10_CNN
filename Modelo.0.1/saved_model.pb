ЈЕ
иЈ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
ћ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628пы
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:
*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:
*
dtype0

Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes
:	
*
dtype0

Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes
:	
*
dtype0

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:*
dtype0

Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel* 
_output_shapes
:
*
dtype0

Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel* 
_output_shapes
:
*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:*
dtype0

Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
 *
dtype0

Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
 *
dtype0

!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*2
shared_name#!Adam/v/batch_normalization_1/beta

5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
:(*
dtype0

!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*2
shared_name#!Adam/m/batch_normalization_1/beta

5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
:(*
dtype0

"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*3
shared_name$"Adam/v/batch_normalization_1/gamma

6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
:(*
dtype0

"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*3
shared_name$"Adam/m/batch_normalization_1/gamma

6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
:(*
dtype0

Adam/v/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/v/conv2d_1/bias
y
(Adam/v/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/bias*
_output_shapes
:(*
dtype0

Adam/m/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/m/conv2d_1/bias
y
(Adam/m/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/bias*
_output_shapes
:(*
dtype0

Adam/v/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*'
shared_nameAdam/v/conv2d_1/kernel

*Adam/v/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/kernel*&
_output_shapes
:((*
dtype0

Adam/m/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((*'
shared_nameAdam/m/conv2d_1/kernel

*Adam/m/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/kernel*&
_output_shapes
:((*
dtype0

Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*0
shared_name!Adam/v/batch_normalization/beta

3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
:(*
dtype0

Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*0
shared_name!Adam/m/batch_normalization/beta

3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
:(*
dtype0

 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" Adam/v/batch_normalization/gamma

4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
:(*
dtype0

 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" Adam/m/batch_normalization/gamma

4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
:(*
dtype0
|
Adam/v/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/v/conv2d/bias
u
&Adam/v/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/bias*
_output_shapes
:(*
dtype0
|
Adam/m/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/m/conv2d/bias
u
&Adam/m/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/bias*
_output_shapes
:(*
dtype0

Adam/v/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/v/conv2d/kernel

(Adam/v/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/kernel*&
_output_shapes
:(*
dtype0

Adam/m/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/m/conv2d/kernel

(Adam/m/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/kernel*&
_output_shapes
:(*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
 *
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:(*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:(*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:(*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:(*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:(*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:((* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:((*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:(*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:(*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:(*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:(*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:(*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:(*
dtype0

serving_default_conv2d_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ  *
dtype0*$
shape:џџџџџџџџџ  
ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_195178

NoOpNoOp
Щp
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*p
valueњoBїo B№o
 
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
е
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%axis
	&gamma
'beta
(moving_mean
)moving_variance*

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
Ш
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
 8_jit_compiled_convolution_op*
е
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance*

D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
І
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
Ѕ
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator* 
І
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias*
Ѕ
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
m_random_generator* 
І
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias*

0
1
&2
'3
(4
)5
66
77
@8
A9
B10
C11
V12
W13
e14
f15
t16
u17*
j
0
1
&2
'3
64
75
@6
A7
V8
W9
e10
f11
t12
u13*
* 
А
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

{trace_0
|trace_1* 

}trace_0
~trace_1* 
* 


_variables
_iterations
_learning_rate
_index_dict

_momentums
_velocities
_update_step_xla*

serving_default* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
&0
'1
(2
)3*

&0
'1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

60
71*

60
71*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Ѓtrace_0* 

Єtrace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
@0
A1
B2
C3*

@0
A1*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

Њtrace_0
Ћtrace_1* 

Ќtrace_0
­trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 
* 
* 
* 

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

Кtrace_0* 

Лtrace_0* 

V0
W1*

V0
W1*
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

Сtrace_0* 

Тtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
* 

e0
f1*

e0
f1*
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

иtrace_0
йtrace_1* 

кtrace_0
лtrace_1* 
* 

t0
u1*

t0
u1*
* 

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

сtrace_0* 

тtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
(0
)1
B2
C3*
Z
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
11*

у0
ф1*
* 
* 
* 
* 
* 
* 
џ
0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
№12
ё13
ђ14
ѓ15
є16
ѕ17
і18
ї19
ј20
љ21
њ22
ћ23
ќ24
§25
ў26
џ27
28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
х0
ч1
щ2
ы3
э4
я5
ё6
ѓ7
ѕ8
ї9
љ10
ћ11
§12
џ13*
x
ц0
ш1
ъ2
ь3
ю4
№5
ђ6
є7
і8
ј9
њ10
ќ11
ў12
13*
* 
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

B0
C1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
_Y
VARIABLE_VALUEAdam/m/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
н
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_rateAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotal_1count_1totalcountConst*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_195821
и
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_rateAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotal_1count_1totalcount*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_195986Ћђ

я

&__inference_dense_layer_call_fn_195382

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_194855p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195378:&"
 
_user_specified_name195376:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Д
§
D__inference_conv2d_1_layer_call_and_return_conditional_losses_195290

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ(S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_194706

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
м
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_195467

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_194747

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs


)__inference_conv2d_1_layer_call_fn_195279

inputs!
unknown:((
	unknown_0:(
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_194822w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ(<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195275:&"
 
_user_specified_name195273:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
д

ѕ
A__inference_dense_layer_call_and_return_conditional_losses_194855

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Д
J
.__inference_max_pooling2d_layer_call_fn_195265

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_194706
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
a
C__inference_dropout_layer_call_and_return_conditional_losses_195420

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
L
0__inference_max_pooling2d_1_layer_call_fn_195357

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_194778
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
D
(__inference_flatten_layer_call_fn_195367

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_194843a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ(:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
фj

!__inference__wrapped_model_194639
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource:(?
1sequential_conv2d_biasadd_readvariableop_resource:(D
6sequential_batch_normalization_readvariableop_resource:(F
8sequential_batch_normalization_readvariableop_1_resource:(U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:(W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:(L
2sequential_conv2d_1_conv2d_readvariableop_resource:((A
3sequential_conv2d_1_biasadd_readvariableop_resource:(F
8sequential_batch_normalization_1_readvariableop_resource:(H
:sequential_batch_normalization_1_readvariableop_1_resource:(W
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:(Y
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:(C
/sequential_dense_matmul_readvariableop_resource:
 ?
0sequential_dense_biasadd_readvariableop_resource:	E
1sequential_dense_1_matmul_readvariableop_resource:
A
2sequential_dense_1_biasadd_readvariableop_resource:	D
1sequential_dense_2_matmul_readvariableop_resource:	
@
2sequential_dense_2_biasadd_readvariableop_resource:

identityЂ>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpЂ@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ-sequential/batch_normalization/ReadVariableOpЂ/sequential/batch_normalization/ReadVariableOp_1Ђ@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂBsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ/sequential/batch_normalization_1/ReadVariableOpЂ1sequential/batch_normalization_1/ReadVariableOp_1Ђ(sequential/conv2d/BiasAdd/ReadVariableOpЂ'sequential/conv2d/Conv2D/ReadVariableOpЂ*sequential/conv2d_1/BiasAdd/ReadVariableOpЂ)sequential/conv2d_1/Conv2D/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpЂ)sequential/dense_2/BiasAdd/ReadVariableOpЂ(sequential/dense_2/MatMul/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0Ф
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Г
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(|
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ( 
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:(*
dtype0Є
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:(*
dtype0Т
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0Ц
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0я
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3$sequential/conv2d/Relu:activations:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
is_training( Э
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ(*
ksize
*
paddingVALID*
strides
Є
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0х
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Й
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(Є
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:(*
dtype0Ј
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:(*
dtype0Ц
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0Ъ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0ћ
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_1/Relu:activations:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
is_training( б
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ(*
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ј
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0Љ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Њ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ў
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџw
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential/dropout_1/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0Џ
sequential/dense_2/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Џ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
|
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
s
IdentityIdentity$sequential/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ  : : : : : : : : : : : : : : : : : : 2
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
/
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameconv2d_input

О
O__inference_batch_normalization_layer_call_and_return_conditional_losses_194657

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs
ю<

F__inference_sequential_layer_call_and_return_conditional_losses_194982
conv2d_input'
conv2d_194923:(
conv2d_194925:((
batch_normalization_194928:((
batch_normalization_194930:((
batch_normalization_194932:((
batch_normalization_194934:()
conv2d_1_194938:((
conv2d_1_194940:(*
batch_normalization_1_194943:(*
batch_normalization_1_194945:(*
batch_normalization_1_194947:(*
batch_normalization_1_194949:( 
dense_194954:
 
dense_194956:	"
dense_1_194965:

dense_1_194967:	!
dense_2_194976:	

dense_2_194978:

identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallі
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_194923conv2d_194925*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_194796
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_194928batch_normalization_194930batch_normalization_194932batch_normalization_194934*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_194675ј
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_194706
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_194938conv2d_1_194940*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_194822
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_194943batch_normalization_1_194945batch_normalization_1_194947batch_normalization_1_194949*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_194747ў
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_194778й
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_194843џ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_194954dense_194956*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_194855з
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_194963
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_194965dense_1_194967*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_194884н
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_194974
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_194976dense_2_194978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_194913w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ  : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:&"
 
_user_specified_name194978:&"
 
_user_specified_name194976:&"
 
_user_specified_name194967:&"
 
_user_specified_name194965:&"
 
_user_specified_name194956:&"
 
_user_specified_name194954:&"
 
_user_specified_name194949:&"
 
_user_specified_name194947:&
"
 
_user_specified_name194945:&	"
 
_user_specified_name194943:&"
 
_user_specified_name194940:&"
 
_user_specified_name194938:&"
 
_user_specified_name194934:&"
 
_user_specified_name194932:&"
 
_user_specified_name194930:&"
 
_user_specified_name194928:&"
 
_user_specified_name194925:&"
 
_user_specified_name194923:] Y
/
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameconv2d_input
Э
a
(__inference_dropout_layer_call_fn_195398

inputs
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_194872p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


b
C__inference_dropout_layer_call_and_return_conditional_losses_195415

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
м
+__inference_sequential_layer_call_fn_195023
conv2d_input!
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:(
	unknown_8:(
	unknown_9:(

unknown_10:(

unknown_11:
 

unknown_12:	

unknown_13:


unknown_14:	

unknown_15:	


unknown_16:

identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_194920o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195019:&"
 
_user_specified_name195017:&"
 
_user_specified_name195015:&"
 
_user_specified_name195013:&"
 
_user_specified_name195011:&"
 
_user_specified_name195009:&"
 
_user_specified_name195007:&"
 
_user_specified_name195005:&
"
 
_user_specified_name195003:&	"
 
_user_specified_name195001:&"
 
_user_specified_name194999:&"
 
_user_specified_name194997:&"
 
_user_specified_name194995:&"
 
_user_specified_name194993:&"
 
_user_specified_name194991:&"
 
_user_specified_name194989:&"
 
_user_specified_name194987:&"
 
_user_specified_name194985:] Y
/
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameconv2d_input
ж

ї
C__inference_dense_1_layer_call_and_return_conditional_losses_195440

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_195270

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_194778

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х
_
C__inference_flatten_layer_call_and_return_conditional_losses_194843

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ(:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
 

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_194901

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

е
$__inference_signature_wrapper_195178
conv2d_input!
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:(
	unknown_8:(
	unknown_9:(

unknown_10:(

unknown_11:
 

unknown_12:	

unknown_13:


unknown_14:	

unknown_15:	


unknown_16:

identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_194639o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195174:&"
 
_user_specified_name195172:&"
 
_user_specified_name195170:&"
 
_user_specified_name195168:&"
 
_user_specified_name195166:&"
 
_user_specified_name195164:&"
 
_user_specified_name195162:&"
 
_user_specified_name195160:&
"
 
_user_specified_name195158:&	"
 
_user_specified_name195156:&"
 
_user_specified_name195154:&"
 
_user_specified_name195152:&"
 
_user_specified_name195150:&"
 
_user_specified_name195148:&"
 
_user_specified_name195146:&"
 
_user_specified_name195144:&"
 
_user_specified_name195142:&"
 
_user_specified_name195140:] Y
/
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameconv2d_input
к
a
C__inference_dropout_layer_call_and_return_conditional_losses_194963

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г

ѕ
C__inference_dense_2_layer_call_and_return_conditional_losses_195487

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д
§
D__inference_conv2d_1_layer_call_and_return_conditional_losses_194822

inputs8
conv2d_readvariableop_resource:((-
biasadd_readvariableop_resource:(
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:((*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ(S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Х
_
C__inference_flatten_layer_call_and_return_conditional_losses_195373

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ(:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
д?
т
F__inference_sequential_layer_call_and_return_conditional_losses_194920
conv2d_input'
conv2d_194797:(
conv2d_194799:((
batch_normalization_194802:((
batch_normalization_194804:((
batch_normalization_194806:((
batch_normalization_194808:()
conv2d_1_194823:((
conv2d_1_194825:(*
batch_normalization_1_194828:(*
batch_normalization_1_194830:(*
batch_normalization_1_194832:(*
batch_normalization_1_194834:( 
dense_194856:
 
dense_194858:	"
dense_1_194885:

dense_1_194887:	!
dense_2_194914:	

dense_2_194916:

identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallі
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_194797conv2d_194799*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_194796џ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_194802batch_normalization_194804batch_normalization_194806batch_normalization_194808*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_194657ј
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_194706
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_194823conv2d_1_194825*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_194822
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_194828batch_normalization_1_194830batch_normalization_1_194832batch_normalization_1_194834*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_194729ў
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_194778й
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_194843џ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_194856dense_194858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_194855ч
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_194872
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_194885dense_1_194887*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_194884
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_194901
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_194914dense_2_194916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_194913w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
ю
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ  : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:&"
 
_user_specified_name194916:&"
 
_user_specified_name194914:&"
 
_user_specified_name194887:&"
 
_user_specified_name194885:&"
 
_user_specified_name194858:&"
 
_user_specified_name194856:&"
 
_user_specified_name194834:&"
 
_user_specified_name194832:&
"
 
_user_specified_name194830:&	"
 
_user_specified_name194828:&"
 
_user_specified_name194825:&"
 
_user_specified_name194823:&"
 
_user_specified_name194808:&"
 
_user_specified_name194806:&"
 
_user_specified_name194804:&"
 
_user_specified_name194802:&"
 
_user_specified_name194799:&"
 
_user_specified_name194797:] Y
/
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameconv2d_input

D
(__inference_dropout_layer_call_fn_195403

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_194963a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
м
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_194974

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

O__inference_batch_normalization_layer_call_and_return_conditional_losses_194675

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs


б
6__inference_batch_normalization_1_layer_call_fn_195316

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_194747
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195312:&"
 
_user_specified_name195310:&"
 
_user_specified_name195308:&"
 
_user_specified_name195306:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs


Я
4__inference_batch_normalization_layer_call_fn_195211

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_194657
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195207:&"
 
_user_specified_name195205:&"
 
_user_specified_name195203:&"
 
_user_specified_name195201:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs
В
ћ
B__inference_conv2d_layer_call_and_return_conditional_losses_195198

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ(S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs


Я
4__inference_batch_normalization_layer_call_fn_195224

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_194675
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195220:&"
 
_user_specified_name195218:&"
 
_user_specified_name195216:&"
 
_user_specified_name195214:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs
ёё
!
"__inference__traced_restore_195986
file_prefix8
assignvariableop_conv2d_kernel:(,
assignvariableop_1_conv2d_bias:(:
,assignvariableop_2_batch_normalization_gamma:(9
+assignvariableop_3_batch_normalization_beta:(@
2assignvariableop_4_batch_normalization_moving_mean:(D
6assignvariableop_5_batch_normalization_moving_variance:(<
"assignvariableop_6_conv2d_1_kernel:((.
 assignvariableop_7_conv2d_1_bias:(<
.assignvariableop_8_batch_normalization_1_gamma:(;
-assignvariableop_9_batch_normalization_1_beta:(C
5assignvariableop_10_batch_normalization_1_moving_mean:(G
9assignvariableop_11_batch_normalization_1_moving_variance:(4
 assignvariableop_12_dense_kernel:
 -
assignvariableop_13_dense_bias:	6
"assignvariableop_14_dense_1_kernel:
/
 assignvariableop_15_dense_1_bias:	5
"assignvariableop_16_dense_2_kernel:	
.
 assignvariableop_17_dense_2_bias:
'
assignvariableop_18_iteration:	 +
!assignvariableop_19_learning_rate: B
(assignvariableop_20_adam_m_conv2d_kernel:(B
(assignvariableop_21_adam_v_conv2d_kernel:(4
&assignvariableop_22_adam_m_conv2d_bias:(4
&assignvariableop_23_adam_v_conv2d_bias:(B
4assignvariableop_24_adam_m_batch_normalization_gamma:(B
4assignvariableop_25_adam_v_batch_normalization_gamma:(A
3assignvariableop_26_adam_m_batch_normalization_beta:(A
3assignvariableop_27_adam_v_batch_normalization_beta:(D
*assignvariableop_28_adam_m_conv2d_1_kernel:((D
*assignvariableop_29_adam_v_conv2d_1_kernel:((6
(assignvariableop_30_adam_m_conv2d_1_bias:(6
(assignvariableop_31_adam_v_conv2d_1_bias:(D
6assignvariableop_32_adam_m_batch_normalization_1_gamma:(D
6assignvariableop_33_adam_v_batch_normalization_1_gamma:(C
5assignvariableop_34_adam_m_batch_normalization_1_beta:(C
5assignvariableop_35_adam_v_batch_normalization_1_beta:(;
'assignvariableop_36_adam_m_dense_kernel:
 ;
'assignvariableop_37_adam_v_dense_kernel:
 4
%assignvariableop_38_adam_m_dense_bias:	4
%assignvariableop_39_adam_v_dense_bias:	=
)assignvariableop_40_adam_m_dense_1_kernel:
=
)assignvariableop_41_adam_v_dense_1_kernel:
6
'assignvariableop_42_adam_m_dense_1_bias:	6
'assignvariableop_43_adam_v_dense_1_bias:	<
)assignvariableop_44_adam_m_dense_2_kernel:	
<
)assignvariableop_45_adam_v_dense_2_kernel:	
5
'assignvariableop_46_adam_m_dense_2_bias:
5
'assignvariableop_47_adam_v_dense_2_bias:
%
assignvariableop_48_total_1: %
assignvariableop_49_count_1: #
assignvariableop_50_total: #
assignvariableop_51_count: 
identity_53ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9я
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*
valueB5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Њ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapesз
д:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_2_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_18AssignVariableOpassignvariableop_18_iterationIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_m_conv2d_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_v_conv2d_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_m_conv2d_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adam_v_conv2d_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_m_batch_normalization_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_v_batch_normalization_gammaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_m_batch_normalization_betaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_v_batch_normalization_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_m_conv2d_1_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_v_conv2d_1_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_m_conv2d_1_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_v_conv2d_1_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_m_batch_normalization_1_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_v_batch_normalization_1_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_m_batch_normalization_1_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_v_batch_normalization_1_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_dense_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_dense_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_m_dense_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_v_dense_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_m_dense_1_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_v_dense_1_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_m_dense_1_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_v_dense_1_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_m_dense_2_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_v_dense_2_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_m_dense_2_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_v_dense_2_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_48AssignVariableOpassignvariableop_48_total_1Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_49AssignVariableOpassignvariableop_49_count_1Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: 	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%4!

_user_specified_namecount:%3!

_user_specified_nametotal:'2#
!
_user_specified_name	count_1:'1#
!
_user_specified_name	total_1:30/
-
_user_specified_nameAdam/v/dense_2/bias:3//
-
_user_specified_nameAdam/m/dense_2/bias:5.1
/
_user_specified_nameAdam/v/dense_2/kernel:5-1
/
_user_specified_nameAdam/m/dense_2/kernel:3,/
-
_user_specified_nameAdam/v/dense_1/bias:3+/
-
_user_specified_nameAdam/m/dense_1/bias:5*1
/
_user_specified_nameAdam/v/dense_1/kernel:5)1
/
_user_specified_nameAdam/m/dense_1/kernel:1(-
+
_user_specified_nameAdam/v/dense/bias:1'-
+
_user_specified_nameAdam/m/dense/bias:3&/
-
_user_specified_nameAdam/v/dense/kernel:3%/
-
_user_specified_nameAdam/m/dense/kernel:A$=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A#=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B">
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B!>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:4 0
.
_user_specified_nameAdam/v/conv2d_1/bias:40
.
_user_specified_nameAdam/m/conv2d_1/bias:62
0
_user_specified_nameAdam/v/conv2d_1/kernel:62
0
_user_specified_nameAdam/m/conv2d_1/kernel:?;
9
_user_specified_name!Adam/v/batch_normalization/beta:?;
9
_user_specified_name!Adam/m/batch_normalization/beta:@<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@<
:
_user_specified_name" Adam/m/batch_normalization/gamma:2.
,
_user_specified_nameAdam/v/conv2d/bias:2.
,
_user_specified_nameAdam/m/conv2d/bias:40
.
_user_specified_nameAdam/v/conv2d/kernel:40
.
_user_specified_nameAdam/m/conv2d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:-)
'
_user_specified_nameconv2d_1/bias:/+
)
_user_specified_nameconv2d_1/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:+'
%
_user_specified_nameconv2d/bias:-)
'
_user_specified_nameconv2d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
а

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195352

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs


'__inference_conv2d_layer_call_fn_195187

inputs!
unknown:(
	unknown_0:(
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_194796w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ(<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195183:&"
 
_user_specified_name195181:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Ѓ
F
*__inference_dropout_1_layer_call_fn_195450

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_194974a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я

(__inference_dense_2_layer_call_fn_195476

inputs
unknown:	

	unknown_0:

identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_194913o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195472:&"
 
_user_specified_name195470:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

O__inference_batch_normalization_layer_call_and_return_conditional_losses_195260

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs
 

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_195462

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_195362

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѓ

(__inference_dense_1_layer_call_fn_195429

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_194884p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195425:&"
 
_user_specified_name195423:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
ћ
B__inference_conv2d_layer_call_and_return_conditional_losses_194796

inputs8
conv2d_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ(S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
ж

ї
C__inference_dense_1_layer_call_and_return_conditional_losses_194884

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б
c
*__inference_dropout_1_layer_call_fn_195445

inputs
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_194901p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г

ѕ
C__inference_dense_2_layer_call_and_return_conditional_losses_194913

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


b
C__inference_dropout_layer_call_and_return_conditional_losses_194872

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


б
6__inference_batch_normalization_1_layer_call_fn_195303

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_194729
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195299:&"
 
_user_specified_name195297:&"
 
_user_specified_name195295:&"
 
_user_specified_name195293:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs

О
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195242

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs
Х
Ѕ0
__inference__traced_save_195821
file_prefix>
$read_disablecopyonread_conv2d_kernel:(2
$read_1_disablecopyonread_conv2d_bias:(@
2read_2_disablecopyonread_batch_normalization_gamma:(?
1read_3_disablecopyonread_batch_normalization_beta:(F
8read_4_disablecopyonread_batch_normalization_moving_mean:(J
<read_5_disablecopyonread_batch_normalization_moving_variance:(B
(read_6_disablecopyonread_conv2d_1_kernel:((4
&read_7_disablecopyonread_conv2d_1_bias:(B
4read_8_disablecopyonread_batch_normalization_1_gamma:(A
3read_9_disablecopyonread_batch_normalization_1_beta:(I
;read_10_disablecopyonread_batch_normalization_1_moving_mean:(M
?read_11_disablecopyonread_batch_normalization_1_moving_variance:(:
&read_12_disablecopyonread_dense_kernel:
 3
$read_13_disablecopyonread_dense_bias:	<
(read_14_disablecopyonread_dense_1_kernel:
5
&read_15_disablecopyonread_dense_1_bias:	;
(read_16_disablecopyonread_dense_2_kernel:	
4
&read_17_disablecopyonread_dense_2_bias:
-
#read_18_disablecopyonread_iteration:	 1
'read_19_disablecopyonread_learning_rate: H
.read_20_disablecopyonread_adam_m_conv2d_kernel:(H
.read_21_disablecopyonread_adam_v_conv2d_kernel:(:
,read_22_disablecopyonread_adam_m_conv2d_bias:(:
,read_23_disablecopyonread_adam_v_conv2d_bias:(H
:read_24_disablecopyonread_adam_m_batch_normalization_gamma:(H
:read_25_disablecopyonread_adam_v_batch_normalization_gamma:(G
9read_26_disablecopyonread_adam_m_batch_normalization_beta:(G
9read_27_disablecopyonread_adam_v_batch_normalization_beta:(J
0read_28_disablecopyonread_adam_m_conv2d_1_kernel:((J
0read_29_disablecopyonread_adam_v_conv2d_1_kernel:((<
.read_30_disablecopyonread_adam_m_conv2d_1_bias:(<
.read_31_disablecopyonread_adam_v_conv2d_1_bias:(J
<read_32_disablecopyonread_adam_m_batch_normalization_1_gamma:(J
<read_33_disablecopyonread_adam_v_batch_normalization_1_gamma:(I
;read_34_disablecopyonread_adam_m_batch_normalization_1_beta:(I
;read_35_disablecopyonread_adam_v_batch_normalization_1_beta:(A
-read_36_disablecopyonread_adam_m_dense_kernel:
 A
-read_37_disablecopyonread_adam_v_dense_kernel:
 :
+read_38_disablecopyonread_adam_m_dense_bias:	:
+read_39_disablecopyonread_adam_v_dense_bias:	C
/read_40_disablecopyonread_adam_m_dense_1_kernel:
C
/read_41_disablecopyonread_adam_v_dense_1_kernel:
<
-read_42_disablecopyonread_adam_m_dense_1_bias:	<
-read_43_disablecopyonread_adam_v_dense_1_bias:	B
/read_44_disablecopyonread_adam_m_dense_2_kernel:	
B
/read_45_disablecopyonread_adam_v_dense_2_kernel:	
;
-read_46_disablecopyonread_adam_m_dense_2_bias:
;
-read_47_disablecopyonread_adam_v_dense_2_bias:
+
!read_48_disablecopyonread_total_1: +
!read_49_disablecopyonread_count_1: )
read_50_disablecopyonread_total: )
read_51_disablecopyonread_count: 
savev2_const
identity_105ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 Ј
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:(*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:(i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:(x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
  
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_2/DisableCopyOnReadDisableCopyOnRead2read_2_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Ў
Read_2/ReadVariableOpReadVariableOp2read_2_disablecopyonread_batch_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 ­
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_batch_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_4/DisableCopyOnReadDisableCopyOnRead8read_4_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 Д
Read_4/ReadVariableOpReadVariableOp8read_4_disablecopyonread_batch_normalization_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_5/DisableCopyOnReadDisableCopyOnRead<read_5_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 И
Read_5/ReadVariableOpReadVariableOp<read_5_disablecopyonread_batch_normalization_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:(|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 А
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:((*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:((m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:((z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 А
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 Џ
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 Й
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:({
Read_12/DisableCopyOnReadDisableCopyOnRead&read_12_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_12/ReadVariableOpReadVariableOp&read_12_disablecopyonread_dense_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 y
Read_13/DisableCopyOnReadDisableCopyOnRead$read_13_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_13/ReadVariableOpReadVariableOp$read_13_disablecopyonread_dense_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_dense_1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
{
Read_15/DisableCopyOnReadDisableCopyOnRead&read_15_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_15/ReadVariableOpReadVariableOp&read_15_disablecopyonread_dense_1_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_dense_2_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	
{
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 Є
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_dense_2_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
x
Read_18/DisableCopyOnReadDisableCopyOnRead#read_18_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_18/ReadVariableOpReadVariableOp#read_18_disablecopyonread_iteration^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_learning_rate^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_m_conv2d_kernel"/device:CPU:0*
_output_shapes
 И
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_m_conv2d_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:(*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:(m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:(
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_v_conv2d_kernel"/device:CPU:0*
_output_shapes
 И
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_v_conv2d_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:(*
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:(m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
:(
Read_22/DisableCopyOnReadDisableCopyOnRead,read_22_disablecopyonread_adam_m_conv2d_bias"/device:CPU:0*
_output_shapes
 Њ
Read_22/ReadVariableOpReadVariableOp,read_22_disablecopyonread_adam_m_conv2d_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_23/DisableCopyOnReadDisableCopyOnRead,read_23_disablecopyonread_adam_v_conv2d_bias"/device:CPU:0*
_output_shapes
 Њ
Read_23/ReadVariableOpReadVariableOp,read_23_disablecopyonread_adam_v_conv2d_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_24/DisableCopyOnReadDisableCopyOnRead:read_24_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 И
Read_24/ReadVariableOpReadVariableOp:read_24_disablecopyonread_adam_m_batch_normalization_gamma^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_25/DisableCopyOnReadDisableCopyOnRead:read_25_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 И
Read_25/ReadVariableOpReadVariableOp:read_25_disablecopyonread_adam_v_batch_normalization_gamma^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_26/DisableCopyOnReadDisableCopyOnRead9read_26_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 З
Read_26/ReadVariableOpReadVariableOp9read_26_disablecopyonread_adam_m_batch_normalization_beta^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_27/DisableCopyOnReadDisableCopyOnRead9read_27_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 З
Read_27/ReadVariableOpReadVariableOp9read_27_disablecopyonread_adam_v_batch_normalization_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_28/DisableCopyOnReadDisableCopyOnRead0read_28_disablecopyonread_adam_m_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 К
Read_28/ReadVariableOpReadVariableOp0read_28_disablecopyonread_adam_m_conv2d_1_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:((*
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:((m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
:((
Read_29/DisableCopyOnReadDisableCopyOnRead0read_29_disablecopyonread_adam_v_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 К
Read_29/ReadVariableOpReadVariableOp0read_29_disablecopyonread_adam_v_conv2d_1_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:((*
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:((m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
:((
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_m_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_m_conv2d_1_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_31/DisableCopyOnReadDisableCopyOnRead.read_31_disablecopyonread_adam_v_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_31/ReadVariableOpReadVariableOp.read_31_disablecopyonread_adam_v_conv2d_1_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_32/DisableCopyOnReadDisableCopyOnRead<read_32_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 К
Read_32/ReadVariableOpReadVariableOp<read_32_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_33/DisableCopyOnReadDisableCopyOnRead<read_33_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 К
Read_33/ReadVariableOpReadVariableOp<read_33_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_34/DisableCopyOnReadDisableCopyOnRead;read_34_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 Й
Read_34/ReadVariableOpReadVariableOp;read_34_disablecopyonread_adam_m_batch_normalization_1_beta^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_35/DisableCopyOnReadDisableCopyOnRead;read_35_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 Й
Read_35/ReadVariableOpReadVariableOp;read_35_disablecopyonread_adam_v_batch_normalization_1_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_36/DisableCopyOnReadDisableCopyOnRead-read_36_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 Б
Read_36/ReadVariableOpReadVariableOp-read_36_disablecopyonread_adam_m_dense_kernel^Read_36/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 
Read_37/DisableCopyOnReadDisableCopyOnRead-read_37_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 Б
Read_37/ReadVariableOpReadVariableOp-read_37_disablecopyonread_adam_v_dense_kernel^Read_37/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 
Read_38/DisableCopyOnReadDisableCopyOnRead+read_38_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 Њ
Read_38/ReadVariableOpReadVariableOp+read_38_disablecopyonread_adam_m_dense_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_39/DisableCopyOnReadDisableCopyOnRead+read_39_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 Њ
Read_39/ReadVariableOpReadVariableOp+read_39_disablecopyonread_adam_v_dense_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 Г
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_adam_m_dense_1_kernel^Read_40/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_41/DisableCopyOnReadDisableCopyOnRead/read_41_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 Г
Read_41/ReadVariableOpReadVariableOp/read_41_disablecopyonread_adam_v_dense_1_kernel^Read_41/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_42/DisableCopyOnReadDisableCopyOnRead-read_42_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_42/ReadVariableOpReadVariableOp-read_42_disablecopyonread_adam_m_dense_1_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_43/DisableCopyOnReadDisableCopyOnRead-read_43_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_43/ReadVariableOpReadVariableOp-read_43_disablecopyonread_adam_v_dense_1_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 В
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_adam_m_dense_2_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0p
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:	

Read_45/DisableCopyOnReadDisableCopyOnRead/read_45_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 В
Read_45/ReadVariableOpReadVariableOp/read_45_disablecopyonread_adam_v_dense_2_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0p
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:	

Read_46/DisableCopyOnReadDisableCopyOnRead-read_46_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_46/ReadVariableOpReadVariableOp-read_46_disablecopyonread_adam_m_dense_2_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:

Read_47/DisableCopyOnReadDisableCopyOnRead-read_47_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_47/ReadVariableOpReadVariableOp-read_47_disablecopyonread_adam_v_dense_2_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_48/DisableCopyOnReadDisableCopyOnRead!read_48_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_48/ReadVariableOpReadVariableOp!read_48_disablecopyonread_total_1^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_49/DisableCopyOnReadDisableCopyOnRead!read_49_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_49/ReadVariableOpReadVariableOp!read_49_disablecopyonread_count_1^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_50/DisableCopyOnReadDisableCopyOnReadread_50_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_50/ReadVariableOpReadVariableOpread_50_disablecopyonread_total^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_51/DisableCopyOnReadDisableCopyOnReadread_51_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_51/ReadVariableOpReadVariableOpread_51_disablecopyonread_count^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: ь
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*
valueB5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *C
dtypes9
725	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_104Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_105IdentityIdentity_104:output:0^NoOp*
T0*
_output_shapes
: у
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_105Identity_105:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=59

_output_shapes
: 

_user_specified_nameConst:%4!

_user_specified_namecount:%3!

_user_specified_nametotal:'2#
!
_user_specified_name	count_1:'1#
!
_user_specified_name	total_1:30/
-
_user_specified_nameAdam/v/dense_2/bias:3//
-
_user_specified_nameAdam/m/dense_2/bias:5.1
/
_user_specified_nameAdam/v/dense_2/kernel:5-1
/
_user_specified_nameAdam/m/dense_2/kernel:3,/
-
_user_specified_nameAdam/v/dense_1/bias:3+/
-
_user_specified_nameAdam/m/dense_1/bias:5*1
/
_user_specified_nameAdam/v/dense_1/kernel:5)1
/
_user_specified_nameAdam/m/dense_1/kernel:1(-
+
_user_specified_nameAdam/v/dense/bias:1'-
+
_user_specified_nameAdam/m/dense/bias:3&/
-
_user_specified_nameAdam/v/dense/kernel:3%/
-
_user_specified_nameAdam/m/dense/kernel:A$=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A#=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B">
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B!>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:4 0
.
_user_specified_nameAdam/v/conv2d_1/bias:40
.
_user_specified_nameAdam/m/conv2d_1/bias:62
0
_user_specified_nameAdam/v/conv2d_1/kernel:62
0
_user_specified_nameAdam/m/conv2d_1/kernel:?;
9
_user_specified_name!Adam/v/batch_normalization/beta:?;
9
_user_specified_name!Adam/m/batch_normalization/beta:@<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@<
:
_user_specified_name" Adam/m/batch_normalization/gamma:2.
,
_user_specified_nameAdam/v/conv2d/bias:2.
,
_user_specified_nameAdam/m/conv2d/bias:40
.
_user_specified_nameAdam/v/conv2d/kernel:40
.
_user_specified_nameAdam/m/conv2d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:-)
'
_user_specified_nameconv2d_1/bias:/+
)
_user_specified_nameconv2d_1/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:+'
%
_user_specified_nameconv2d/bias:-)
'
_user_specified_nameconv2d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Р
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_194729

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs

Р
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195334

inputs%
readvariableop_resource:('
readvariableop_1_resource:(6
(fusedbatchnormv3_readvariableop_resource:(8
*fusedbatchnormv3_readvariableop_1_resource:(
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(:(:(:(:(:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(А
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs
Щ
м
+__inference_sequential_layer_call_fn_195064
conv2d_input!
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(#
	unknown_5:((
	unknown_6:(
	unknown_7:(
	unknown_8:(
	unknown_9:(

unknown_10:(

unknown_11:
 

unknown_12:	

unknown_13:


unknown_14:	

unknown_15:	


unknown_16:

identityЂStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_194982o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name195060:&"
 
_user_specified_name195058:&"
 
_user_specified_name195056:&"
 
_user_specified_name195054:&"
 
_user_specified_name195052:&"
 
_user_specified_name195050:&"
 
_user_specified_name195048:&"
 
_user_specified_name195046:&
"
 
_user_specified_name195044:&	"
 
_user_specified_name195042:&"
 
_user_specified_name195040:&"
 
_user_specified_name195038:&"
 
_user_specified_name195036:&"
 
_user_specified_name195034:&"
 
_user_specified_name195032:&"
 
_user_specified_name195030:&"
 
_user_specified_name195028:&"
 
_user_specified_name195026:] Y
/
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameconv2d_input
д

ѕ
A__inference_dense_layer_call_and_return_conditional_losses_195393

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs"ЪL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*М
serving_defaultЈ
M
conv2d_input=
serving_default_conv2d_input:0џџџџџџџџџ  ;
dense_20
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:јЏ
К
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
ъ
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%axis
	&gamma
'beta
(moving_mean
)moving_variance"
_tf_keras_layer
Ѕ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
н
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
 8_jit_compiled_convolution_op"
_tf_keras_layer
ъ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance"
_tf_keras_layer
Ѕ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
М
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator"
_tf_keras_layer
Л
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias"
_tf_keras_layer
М
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
m_random_generator"
_tf_keras_layer
Л
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias"
_tf_keras_layer
І
0
1
&2
'3
(4
)5
66
77
@8
A9
B10
C11
V12
W13
e14
f15
t16
u17"
trackable_list_wrapper

0
1
&2
'3
64
75
@6
A7
V8
W9
e10
f11
t12
u13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Щ
{trace_0
|trace_12
+__inference_sequential_layer_call_fn_195023
+__inference_sequential_layer_call_fn_195064Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z{trace_0z|trace_1
џ
}trace_0
~trace_12Ш
F__inference_sequential_layer_call_and_return_conditional_losses_194920
F__inference_sequential_layer_call_and_return_conditional_losses_194982Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z}trace_0z~trace_1
бBЮ
!__inference__wrapped_model_194639conv2d_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ђ

_variables
_iterations
_learning_rate
_index_dict

_momentums
_velocities
_update_step_xla"
experimentalOptimizer
-
serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
у
trace_02Ф
'__inference_conv2d_layer_call_fn_195187
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
ў
trace_02п
B__inference_conv2d_layer_call_and_return_conditional_losses_195198
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
':%(2conv2d/kernel
:(2conv2d/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
п
trace_0
trace_12Є
4__inference_batch_normalization_layer_call_fn_195211
4__inference_batch_normalization_layer_call_fn_195224Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195242
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195260Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
':%(2batch_normalization/gamma
&:$(2batch_normalization/beta
/:-( (2batch_normalization/moving_mean
3:1( (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_max_pooling2d_layer_call_fn_195265
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_195270
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
х
Ѓtrace_02Ц
)__inference_conv2d_1_layer_call_fn_195279
В
FullArgSpec
args

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
annotationsЊ *
 zЃtrace_0

Єtrace_02с
D__inference_conv2d_1_layer_call_and_return_conditional_losses_195290
В
FullArgSpec
args

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
annotationsЊ *
 zЄtrace_0
):'((2conv2d_1/kernel
:(2conv2d_1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
у
Њtrace_0
Ћtrace_12Ј
6__inference_batch_normalization_1_layer_call_fn_195303
6__inference_batch_normalization_1_layer_call_fn_195316Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0zЋtrace_1

Ќtrace_0
­trace_12о
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195334
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195352Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0z­trace_1
 "
trackable_list_wrapper
):'(2batch_normalization_1/gamma
(:&(2batch_normalization_1/beta
1:/( (2!batch_normalization_1/moving_mean
5:3( (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ь
Гtrace_02Э
0__inference_max_pooling2d_1_layer_call_fn_195357
В
FullArgSpec
args

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
annotationsЊ *
 zГtrace_0

Дtrace_02ш
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_195362
В
FullArgSpec
args

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
annotationsЊ *
 zДtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ф
Кtrace_02Х
(__inference_flatten_layer_call_fn_195367
В
FullArgSpec
args

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
annotationsЊ *
 zКtrace_0
џ
Лtrace_02р
C__inference_flatten_layer_call_and_return_conditional_losses_195373
В
FullArgSpec
args

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
annotationsЊ *
 zЛtrace_0
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
т
Сtrace_02У
&__inference_dense_layer_call_fn_195382
В
FullArgSpec
args

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
annotationsЊ *
 zСtrace_0
§
Тtrace_02о
A__inference_dense_layer_call_and_return_conditional_losses_195393
В
FullArgSpec
args

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
annotationsЊ *
 zТtrace_0
 :
 2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
Л
Шtrace_0
Щtrace_12
(__inference_dropout_layer_call_fn_195398
(__inference_dropout_layer_call_fn_195403Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zШtrace_0zЩtrace_1
ё
Ъtrace_0
Ыtrace_12Ж
C__inference_dropout_layer_call_and_return_conditional_losses_195415
C__inference_dropout_layer_call_and_return_conditional_losses_195420Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0zЫtrace_1
"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ф
бtrace_02Х
(__inference_dense_1_layer_call_fn_195429
В
FullArgSpec
args

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
annotationsЊ *
 zбtrace_0
џ
вtrace_02р
C__inference_dense_1_layer_call_and_return_conditional_losses_195440
В
FullArgSpec
args

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
annotationsЊ *
 zвtrace_0
": 
2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
П
иtrace_0
йtrace_12
*__inference_dropout_1_layer_call_fn_195445
*__inference_dropout_1_layer_call_fn_195450Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zиtrace_0zйtrace_1
ѕ
кtrace_0
лtrace_12К
E__inference_dropout_1_layer_call_and_return_conditional_losses_195462
E__inference_dropout_1_layer_call_and_return_conditional_losses_195467Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0zлtrace_1
"
_generic_user_object
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ф
сtrace_02Х
(__inference_dense_2_layer_call_fn_195476
В
FullArgSpec
args

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
annotationsЊ *
 zсtrace_0
џ
тtrace_02р
C__inference_dense_2_layer_call_and_return_conditional_losses_195487
В
FullArgSpec
args

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
annotationsЊ *
 zтtrace_0
!:	
2dense_2/kernel
:
2dense_2/bias
<
(0
)1
B2
C3"
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
0
у0
ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBѕ
+__inference_sequential_layer_call_fn_195023conv2d_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
+__inference_sequential_layer_call_fn_195064conv2d_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_194920conv2d_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_194982conv2d_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
№12
ё13
ђ14
ѓ15
є16
ѕ17
і18
ї19
ј20
љ21
њ22
ћ23
ќ24
§25
ў26
џ27
28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper

х0
ч1
щ2
ы3
э4
я5
ё6
ѓ7
ѕ8
ї9
љ10
ћ11
§12
џ13"
trackable_list_wrapper

ц0
ш1
ъ2
ь3
ю4
№5
ђ6
є7
і8
ј9
њ10
ќ11
ў12
13"
trackable_list_wrapper
Е2ВЏ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
аBЭ
$__inference_signature_wrapper_195178conv2d_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
бBЮ
'__inference_conv2d_layer_call_fn_195187inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ьBщ
B__inference_conv2d_layer_call_and_return_conditional_losses_195198inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћBј
4__inference_batch_normalization_layer_call_fn_195211inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
4__inference_batch_normalization_layer_call_fn_195224inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195242inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195260inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
иBе
.__inference_max_pooling2d_layer_call_fn_195265inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_195270inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
гBа
)__inference_conv2d_1_layer_call_fn_195279inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_conv2d_1_layer_call_and_return_conditional_losses_195290inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
6__inference_batch_normalization_1_layer_call_fn_195303inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
6__inference_batch_normalization_1_layer_call_fn_195316inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195334inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195352inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
кBз
0__inference_max_pooling2d_1_layer_call_fn_195357inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѕBђ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_195362inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
вBЯ
(__inference_flatten_layer_call_fn_195367inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
эBъ
C__inference_flatten_layer_call_and_return_conditional_losses_195373inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
аBЭ
&__inference_dense_layer_call_fn_195382inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ыBш
A__inference_dense_layer_call_and_return_conditional_losses_195393inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
уBр
(__inference_dropout_layer_call_fn_195398inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
уBр
(__inference_dropout_layer_call_fn_195403inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
C__inference_dropout_layer_call_and_return_conditional_losses_195415inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
C__inference_dropout_layer_call_and_return_conditional_losses_195420inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
вBЯ
(__inference_dense_1_layer_call_fn_195429inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
эBъ
C__inference_dense_1_layer_call_and_return_conditional_losses_195440inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
хBт
*__inference_dropout_1_layer_call_fn_195445inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
хBт
*__inference_dropout_1_layer_call_fn_195450inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
E__inference_dropout_1_layer_call_and_return_conditional_losses_195462inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
E__inference_dropout_1_layer_call_and_return_conditional_losses_195467inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
вBЯ
(__inference_dense_2_layer_call_fn_195476inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
эBъ
C__inference_dense_2_layer_call_and_return_conditional_losses_195487inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
,:*(2Adam/m/conv2d/kernel
,:*(2Adam/v/conv2d/kernel
:(2Adam/m/conv2d/bias
:(2Adam/v/conv2d/bias
,:*(2 Adam/m/batch_normalization/gamma
,:*(2 Adam/v/batch_normalization/gamma
+:)(2Adam/m/batch_normalization/beta
+:)(2Adam/v/batch_normalization/beta
.:,((2Adam/m/conv2d_1/kernel
.:,((2Adam/v/conv2d_1/kernel
 :(2Adam/m/conv2d_1/bias
 :(2Adam/v/conv2d_1/bias
.:,(2"Adam/m/batch_normalization_1/gamma
.:,(2"Adam/v/batch_normalization_1/gamma
-:+(2!Adam/m/batch_normalization_1/beta
-:+(2!Adam/v/batch_normalization_1/beta
%:#
 2Adam/m/dense/kernel
%:#
 2Adam/v/dense/kernel
:2Adam/m/dense/bias
:2Adam/v/dense/bias
':%
2Adam/m/dense_1/kernel
':%
2Adam/v/dense_1/kernel
 :2Adam/m/dense_1/bias
 :2Adam/v/dense_1/bias
&:$	
2Adam/m/dense_2/kernel
&:$	
2Adam/v/dense_2/kernel
:
2Adam/m/dense_2/bias
:
2Adam/v/dense_2/bias
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЌ
!__inference__wrapped_model_194639&'()67@ABCVWeftu=Ђ:
3Ђ0
.+
conv2d_inputџџџџџџџџџ  
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџ
ї
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195334Ё@ABCQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 ї
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_195352Ё@ABCQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 б
6__inference_batch_normalization_1_layer_call_fn_195303@ABCQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(б
6__inference_batch_normalization_1_layer_call_fn_195316@ABCQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(ѕ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195242Ё&'()QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 ѕ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_195260Ё&'()QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
 Я
4__inference_batch_normalization_layer_call_fn_195211&'()QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(Я
4__inference_batch_normalization_layer_call_fn_195224&'()QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ(Л
D__inference_conv2d_1_layer_call_and_return_conditional_losses_195290s677Ђ4
-Ђ*
(%
inputsџџџџџџџџџ(
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ(
 
)__inference_conv2d_1_layer_call_fn_195279h677Ђ4
-Ђ*
(%
inputsџџџџџџџџџ(
Њ ")&
unknownџџџџџџџџџ(Й
B__inference_conv2d_layer_call_and_return_conditional_losses_195198s7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ(
 
'__inference_conv2d_layer_call_fn_195187h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  
Њ ")&
unknownџџџџџџџџџ(Ќ
C__inference_dense_1_layer_call_and_return_conditional_losses_195440eef0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
(__inference_dense_1_layer_call_fn_195429Zef0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџЋ
C__inference_dense_2_layer_call_and_return_conditional_losses_195487dtu0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
(__inference_dense_2_layer_call_fn_195476Ytu0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
Њ
A__inference_dense_layer_call_and_return_conditional_losses_195393eVW0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
&__inference_dense_layer_call_fn_195382ZVW0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџЎ
E__inference_dropout_1_layer_call_and_return_conditional_losses_195462e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 Ў
E__inference_dropout_1_layer_call_and_return_conditional_losses_195467e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
*__inference_dropout_1_layer_call_fn_195445Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџ
*__inference_dropout_1_layer_call_fn_195450Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџЌ
C__inference_dropout_layer_call_and_return_conditional_losses_195415e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 Ќ
C__inference_dropout_layer_call_and_return_conditional_losses_195420e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
(__inference_dropout_layer_call_fn_195398Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџ
(__inference_dropout_layer_call_fn_195403Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџЏ
C__inference_flatten_layer_call_and_return_conditional_losses_195373h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ(
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
(__inference_flatten_layer_call_fn_195367]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ(
Њ ""
unknownџџџџџџџџџ ѕ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_195362ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
0__inference_max_pooling2d_1_layer_call_fn_195357RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_195270ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Э
.__inference_max_pooling2d_layer_call_fn_195265RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџд
F__inference_sequential_layer_call_and_return_conditional_losses_194920&'()67@ABCVWeftuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ  
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 д
F__inference_sequential_layer_call_and_return_conditional_losses_194982&'()67@ABCVWeftuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ  
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 ­
+__inference_sequential_layer_call_fn_195023~&'()67@ABCVWeftuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ  
p

 
Њ "!
unknownџџџџџџџџџ
­
+__inference_sequential_layer_call_fn_195064~&'()67@ABCVWeftuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ  
p 

 
Њ "!
unknownџџџџџџџџџ
П
$__inference_signature_wrapper_195178&'()67@ABCVWeftuMЂJ
Ђ 
CЊ@
>
conv2d_input.+
conv2d_inputџџџџџџџџџ  "1Њ.
,
dense_2!
dense_2џџџџџџџџџ
