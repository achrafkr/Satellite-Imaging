иё:
ЯГ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
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
Л
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

·
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
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
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
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02unknown8ЎИ0
Г
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_8/kernel
|
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*'
_output_shapes
:А*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_16/gamma
К
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_16/beta
И
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_16/moving_mean
Ц
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_16/moving_variance
Ю
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_14/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_14/depthwise_kernel
ж
8separable_conv2d_14/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_14/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_14/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_14/pointwise_kernel
з
8separable_conv2d_14/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_14/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_14/bias
В
,separable_conv2d_14/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_14/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_17/gamma
К
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_17/beta
И
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_17/moving_mean
Ц
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_17/moving_variance
Ю
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_15/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_15/depthwise_kernel
ж
8separable_conv2d_15/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_15/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_15/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_15/pointwise_kernel
з
8separable_conv2d_15/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_15/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_15/bias
В
,separable_conv2d_15/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_15/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_18/gamma
К
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_18/beta
И
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_18/moving_mean
Ц
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_18/moving_variance
Ю
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:А*
dtype0
Д
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_16/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_16/depthwise_kernel
ж
8separable_conv2d_16/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_16/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_16/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_16/pointwise_kernel
з
8separable_conv2d_16/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_16/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_16/bias
В
,separable_conv2d_16/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_16/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_19/gamma
К
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_19/beta
И
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_19/moving_mean
Ц
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_19/moving_variance
Ю
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_17/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_17/depthwise_kernel
ж
8separable_conv2d_17/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_17/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_17/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*5
shared_name&$separable_conv2d_17/pointwise_kernel
з
8separable_conv2d_17/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_17/pointwise_kernel*(
_output_shapes
:АА*
dtype0
Й
separable_conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_17/bias
В
,separable_conv2d_17/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_17/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_20/gamma
К
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_20/beta
И
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_20/moving_mean
Ц
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_20/moving_variance
Ю
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes	
:А*
dtype0
Ж
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_10/kernel

$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:А*
dtype0
н
$separable_conv2d_18/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$separable_conv2d_18/depthwise_kernel
ж
8separable_conv2d_18/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_18/depthwise_kernel*'
_output_shapes
:А*
dtype0
о
$separable_conv2d_18/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А╪*5
shared_name&$separable_conv2d_18/pointwise_kernel
з
8separable_conv2d_18/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_18/pointwise_kernel*(
_output_shapes
:А╪*
dtype0
Й
separable_conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*)
shared_nameseparable_conv2d_18/bias
В
,separable_conv2d_18/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_18/bias*
_output_shapes	
:╪*
dtype0
С
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*-
shared_namebatch_normalization_21/gamma
К
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes	
:╪*
dtype0
П
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*,
shared_namebatch_normalization_21/beta
И
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes	
:╪*
dtype0
Э
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*3
shared_name$"batch_normalization_21/moving_mean
Ц
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes	
:╪*
dtype0
е
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*7
shared_name(&batch_normalization_21/moving_variance
Ю
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes	
:╪*
dtype0
н
$separable_conv2d_19/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*5
shared_name&$separable_conv2d_19/depthwise_kernel
ж
8separable_conv2d_19/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_19/depthwise_kernel*'
_output_shapes
:╪*
dtype0
о
$separable_conv2d_19/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪╪*5
shared_name&$separable_conv2d_19/pointwise_kernel
з
8separable_conv2d_19/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_19/pointwise_kernel*(
_output_shapes
:╪╪*
dtype0
Й
separable_conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*)
shared_nameseparable_conv2d_19/bias
В
,separable_conv2d_19/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_19/bias*
_output_shapes	
:╪*
dtype0
С
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*-
shared_namebatch_normalization_22/gamma
К
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes	
:╪*
dtype0
П
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*,
shared_namebatch_normalization_22/beta
И
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes	
:╪*
dtype0
Э
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*3
shared_name$"batch_normalization_22/moving_mean
Ц
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes	
:╪*
dtype0
е
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*7
shared_name(&batch_normalization_22/moving_variance
Ю
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes	
:╪*
dtype0
Ж
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А╪*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:А╪*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:╪*
dtype0
н
$separable_conv2d_20/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*5
shared_name&$separable_conv2d_20/depthwise_kernel
ж
8separable_conv2d_20/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_20/depthwise_kernel*'
_output_shapes
:╪*
dtype0
о
$separable_conv2d_20/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪А*5
shared_name&$separable_conv2d_20/pointwise_kernel
з
8separable_conv2d_20/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_20/pointwise_kernel*(
_output_shapes
:╪А*
dtype0
Й
separable_conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameseparable_conv2d_20/bias
В
,separable_conv2d_20/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_20/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_23/gamma
К
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_23/beta
И
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_23/moving_mean
Ц
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_23/moving_variance
Ю
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:А*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	А
*
dtype0
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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
С
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/conv2d_8/kernel/m
К
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*'
_output_shapes
:А*
dtype0
Б
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_8/bias/m
z
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_16/gamma/m
Ш
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_16/beta/m
Ц
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_14/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_14/depthwise_kernel/m
┤
?Adam/separable_conv2d_14/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_14/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_14/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_14/pointwise_kernel/m
╡
?Adam/separable_conv2d_14/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_14/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_14/bias/m
Р
3Adam/separable_conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_14/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_17/gamma/m
Ш
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_17/beta/m
Ц
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_15/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_15/depthwise_kernel/m
┤
?Adam/separable_conv2d_15/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_15/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_15/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_15/pointwise_kernel/m
╡
?Adam/separable_conv2d_15/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_15/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_15/bias/m
Р
3Adam/separable_conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_15/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_18/gamma/m
Ш
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_18/beta/m
Ц
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_9/kernel/m
Л
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_9/bias/m
z
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_16/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_16/depthwise_kernel/m
┤
?Adam/separable_conv2d_16/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_16/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_16/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_16/pointwise_kernel/m
╡
?Adam/separable_conv2d_16/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_16/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_16/bias/m
Р
3Adam/separable_conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_16/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_19/gamma/m
Ш
7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_19/beta/m
Ц
6Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/m*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_17/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_17/depthwise_kernel/m
┤
?Adam/separable_conv2d_17/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_17/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_17/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_17/pointwise_kernel/m
╡
?Adam/separable_conv2d_17/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_17/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_17/bias/m
Р
3Adam/separable_conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_17/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_20/gamma/m
Ш
7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_20/beta/m
Ц
6Adam/batch_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_10/kernel/m
Н
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_10/bias/m
|
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_18/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_18/depthwise_kernel/m
┤
?Adam/separable_conv2d_18/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_18/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_18/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А╪*<
shared_name-+Adam/separable_conv2d_18/pointwise_kernel/m
╡
?Adam/separable_conv2d_18/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_18/pointwise_kernel/m*(
_output_shapes
:А╪*
dtype0
Ч
Adam/separable_conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*0
shared_name!Adam/separable_conv2d_18/bias/m
Р
3Adam/separable_conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_18/bias/m*
_output_shapes	
:╪*
dtype0
Я
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*4
shared_name%#Adam/batch_normalization_21/gamma/m
Ш
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes	
:╪*
dtype0
Э
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*3
shared_name$"Adam/batch_normalization_21/beta/m
Ц
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes	
:╪*
dtype0
╗
+Adam/separable_conv2d_19/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*<
shared_name-+Adam/separable_conv2d_19/depthwise_kernel/m
┤
?Adam/separable_conv2d_19/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_19/depthwise_kernel/m*'
_output_shapes
:╪*
dtype0
╝
+Adam/separable_conv2d_19/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪╪*<
shared_name-+Adam/separable_conv2d_19/pointwise_kernel/m
╡
?Adam/separable_conv2d_19/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_19/pointwise_kernel/m*(
_output_shapes
:╪╪*
dtype0
Ч
Adam/separable_conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*0
shared_name!Adam/separable_conv2d_19/bias/m
Р
3Adam/separable_conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_19/bias/m*
_output_shapes	
:╪*
dtype0
Я
#Adam/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*4
shared_name%#Adam/batch_normalization_22/gamma/m
Ш
7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/m*
_output_shapes	
:╪*
dtype0
Э
"Adam/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*3
shared_name$"Adam/batch_normalization_22/beta/m
Ц
6Adam/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/m*
_output_shapes	
:╪*
dtype0
Ф
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А╪*(
shared_nameAdam/conv2d_11/kernel/m
Н
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*(
_output_shapes
:А╪*
dtype0
Г
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*&
shared_nameAdam/conv2d_11/bias/m
|
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes	
:╪*
dtype0
╗
+Adam/separable_conv2d_20/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*<
shared_name-+Adam/separable_conv2d_20/depthwise_kernel/m
┤
?Adam/separable_conv2d_20/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_20/depthwise_kernel/m*'
_output_shapes
:╪*
dtype0
╝
+Adam/separable_conv2d_20/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪А*<
shared_name-+Adam/separable_conv2d_20/pointwise_kernel/m
╡
?Adam/separable_conv2d_20/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_20/pointwise_kernel/m*(
_output_shapes
:╪А*
dtype0
Ч
Adam/separable_conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_20/bias/m
Р
3Adam/separable_conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_20/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_23/gamma/m
Ш
7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_23/beta/m
Ц
6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*&
shared_nameAdam/dense_2/kernel/m
А
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	А
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:
*
dtype0
С
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/conv2d_8/kernel/v
К
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*'
_output_shapes
:А*
dtype0
Б
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_8/bias/v
z
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_16/gamma/v
Ш
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_16/beta/v
Ц
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_14/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_14/depthwise_kernel/v
┤
?Adam/separable_conv2d_14/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_14/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_14/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_14/pointwise_kernel/v
╡
?Adam/separable_conv2d_14/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_14/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_14/bias/v
Р
3Adam/separable_conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_14/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_17/gamma/v
Ш
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_17/beta/v
Ц
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_15/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_15/depthwise_kernel/v
┤
?Adam/separable_conv2d_15/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_15/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_15/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_15/pointwise_kernel/v
╡
?Adam/separable_conv2d_15/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_15/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_15/bias/v
Р
3Adam/separable_conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_15/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_18/gamma/v
Ш
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_18/beta/v
Ц
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_9/kernel/v
Л
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_9/bias/v
z
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_16/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_16/depthwise_kernel/v
┤
?Adam/separable_conv2d_16/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_16/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_16/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_16/pointwise_kernel/v
╡
?Adam/separable_conv2d_16/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_16/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_16/bias/v
Р
3Adam/separable_conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_16/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_19/gamma/v
Ш
7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_19/beta/v
Ц
6Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/v*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_17/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_17/depthwise_kernel/v
┤
?Adam/separable_conv2d_17/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_17/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_17/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*<
shared_name-+Adam/separable_conv2d_17/pointwise_kernel/v
╡
?Adam/separable_conv2d_17/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_17/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Ч
Adam/separable_conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_17/bias/v
Р
3Adam/separable_conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_17/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_20/gamma/v
Ш
7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_20/beta/v
Ц
6Adam/batch_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_10/kernel/v
Н
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_10/bias/v
|
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes	
:А*
dtype0
╗
+Adam/separable_conv2d_18/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/separable_conv2d_18/depthwise_kernel/v
┤
?Adam/separable_conv2d_18/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_18/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
╝
+Adam/separable_conv2d_18/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А╪*<
shared_name-+Adam/separable_conv2d_18/pointwise_kernel/v
╡
?Adam/separable_conv2d_18/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_18/pointwise_kernel/v*(
_output_shapes
:А╪*
dtype0
Ч
Adam/separable_conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*0
shared_name!Adam/separable_conv2d_18/bias/v
Р
3Adam/separable_conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_18/bias/v*
_output_shapes	
:╪*
dtype0
Я
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*4
shared_name%#Adam/batch_normalization_21/gamma/v
Ш
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes	
:╪*
dtype0
Э
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*3
shared_name$"Adam/batch_normalization_21/beta/v
Ц
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes	
:╪*
dtype0
╗
+Adam/separable_conv2d_19/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*<
shared_name-+Adam/separable_conv2d_19/depthwise_kernel/v
┤
?Adam/separable_conv2d_19/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_19/depthwise_kernel/v*'
_output_shapes
:╪*
dtype0
╝
+Adam/separable_conv2d_19/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪╪*<
shared_name-+Adam/separable_conv2d_19/pointwise_kernel/v
╡
?Adam/separable_conv2d_19/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_19/pointwise_kernel/v*(
_output_shapes
:╪╪*
dtype0
Ч
Adam/separable_conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*0
shared_name!Adam/separable_conv2d_19/bias/v
Р
3Adam/separable_conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_19/bias/v*
_output_shapes	
:╪*
dtype0
Я
#Adam/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*4
shared_name%#Adam/batch_normalization_22/gamma/v
Ш
7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/v*
_output_shapes	
:╪*
dtype0
Э
"Adam/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*3
shared_name$"Adam/batch_normalization_22/beta/v
Ц
6Adam/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/v*
_output_shapes	
:╪*
dtype0
Ф
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А╪*(
shared_nameAdam/conv2d_11/kernel/v
Н
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*(
_output_shapes
:А╪*
dtype0
Г
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*&
shared_nameAdam/conv2d_11/bias/v
|
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes	
:╪*
dtype0
╗
+Adam/separable_conv2d_20/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪*<
shared_name-+Adam/separable_conv2d_20/depthwise_kernel/v
┤
?Adam/separable_conv2d_20/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_20/depthwise_kernel/v*'
_output_shapes
:╪*
dtype0
╝
+Adam/separable_conv2d_20/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪А*<
shared_name-+Adam/separable_conv2d_20/pointwise_kernel/v
╡
?Adam/separable_conv2d_20/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/separable_conv2d_20/pointwise_kernel/v*(
_output_shapes
:╪А*
dtype0
Ч
Adam/separable_conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/separable_conv2d_20/bias/v
Р
3Adam/separable_conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_20/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_23/gamma/v
Ш
7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_23/beta/v
Ц
6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes	
:А*
dtype0
З
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*&
shared_nameAdam/dense_2/kernel/v
А
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	А
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
╢ж
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ёе
valueхеBсе B┘е
Н	
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer_with_weights-12
layer-24
layer_with_weights-13
layer-25
layer-26
layer_with_weights-14
layer-27
layer_with_weights-15
layer-28
layer-29
layer_with_weights-16
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"layer_with_weights-18
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-19
&layer-37
'	optimizer
(regularization_losses
)trainable_variables
*	variables
+	keras_api
,
signatures
 

-	keras_api
h

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
Ч
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
И
Edepthwise_kernel
Fpointwise_kernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
Ч
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
R
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
И
Ydepthwise_kernel
Zpointwise_kernel
[bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
Ч
`axis
	agamma
bbeta
cmoving_mean
dmoving_variance
etrainable_variables
fregularization_losses
g	variables
h	keras_api
R
itrainable_variables
jregularization_losses
k	variables
l	keras_api
h

mkernel
nbias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
R
strainable_variables
tregularization_losses
u	variables
v	keras_api
R
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
К
{depthwise_kernel
|pointwise_kernel
}bias
~trainable_variables
regularization_losses
А	variables
Б	keras_api
а
	Вaxis

Гgamma
	Дbeta
Еmoving_mean
Жmoving_variance
Зtrainable_variables
Иregularization_losses
Й	variables
К	keras_api
V
Лtrainable_variables
Мregularization_losses
Н	variables
О	keras_api
П
Пdepthwise_kernel
Рpointwise_kernel
	Сbias
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
а
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ыtrainable_variables
Ьregularization_losses
Э	variables
Ю	keras_api
V
Яtrainable_variables
аregularization_losses
б	variables
в	keras_api
n
гkernel
	дbias
еtrainable_variables
жregularization_losses
з	variables
и	keras_api
V
йtrainable_variables
кregularization_losses
л	variables
м	keras_api
V
нtrainable_variables
оregularization_losses
п	variables
░	keras_api
П
▒depthwise_kernel
▓pointwise_kernel
	│bias
┤trainable_variables
╡regularization_losses
╢	variables
╖	keras_api
а
	╕axis

╣gamma
	║beta
╗moving_mean
╝moving_variance
╜trainable_variables
╛regularization_losses
┐	variables
└	keras_api
V
┴trainable_variables
┬regularization_losses
├	variables
─	keras_api
П
┼depthwise_kernel
╞pointwise_kernel
	╟bias
╚trainable_variables
╔regularization_losses
╩	variables
╦	keras_api
а
	╠axis

═gamma
	╬beta
╧moving_mean
╨moving_variance
╤trainable_variables
╥regularization_losses
╙	variables
╘	keras_api
V
╒trainable_variables
╓regularization_losses
╫	variables
╪	keras_api
n
┘kernel
	┌bias
█trainable_variables
▄regularization_losses
▌	variables
▐	keras_api
V
▀trainable_variables
рregularization_losses
с	variables
т	keras_api
П
уdepthwise_kernel
фpointwise_kernel
	хbias
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
а
	ъaxis

ыgamma
	ьbeta
эmoving_mean
юmoving_variance
яtrainable_variables
Ёregularization_losses
ё	variables
Є	keras_api
V
єtrainable_variables
Їregularization_losses
ї	variables
Ў	keras_api
V
ўtrainable_variables
°regularization_losses
∙	variables
·	keras_api
V
√trainable_variables
№regularization_losses
¤	variables
■	keras_api
n
 kernel
	Аbias
Бtrainable_variables
Вregularization_losses
Г	variables
Д	keras_api
й
	Еiter
Жbeta_1
Зbeta_2

Иdecay
Йlearning_rate.m╬/m╧5m╨6m╤Em╥Fm╙Gm╘Mm╒Nm╓Ym╫Zm╪[m┘am┌bm█mm▄nm▌{m▐|m▀}mр	Гmс	Дmт	Пmу	Рmф	Сmх	Чmц	Шmч	гmш	дmщ	▒mъ	▓mы	│mь	╣mэ	║mю	┼mя	╞mЁ	╟mё	═mЄ	╬mє	┘mЇ	┌mї	уmЎ	фmў	хm°	ыm∙	ьm·	 m√	Аm№.v¤/v■5v 6vАEvБFvВGvГMvДNvЕYvЖZvЗ[vИavЙbvКmvЛnvМ{vН|vО}vП	ГvР	ДvС	ПvТ	РvУ	СvФ	ЧvХ	ШvЦ	гvЧ	дvШ	▒vЩ	▓vЪ	│vЫ	╣vЬ	║vЭ	┼vЮ	╞vЯ	╟vа	═vб	╬vв	┘vг	┌vд	уvе	фvж	хvз	ыvи	ьvй	 vк	Аvл
 
К
.0
/1
52
63
E4
F5
G6
M7
N8
Y9
Z10
[11
a12
b13
m14
n15
{16
|17
}18
Г19
Д20
П21
Р22
С23
Ч24
Ш25
г26
д27
▒28
▓29
│30
╣31
║32
┼33
╞34
╟35
═36
╬37
┘38
┌39
у40
ф41
х42
ы43
ь44
 45
А46
Ф
.0
/1
52
63
74
85
E6
F7
G8
M9
N10
O11
P12
Y13
Z14
[15
a16
b17
c18
d19
m20
n21
{22
|23
}24
Г25
Д26
Е27
Ж28
П29
Р30
С31
Ч32
Ш33
Щ34
Ъ35
г36
д37
▒38
▓39
│40
╣41
║42
╗43
╝44
┼45
╞46
╟47
═48
╬49
╧50
╨51
┘52
┌53
у54
ф55
х56
ы57
ь58
э59
ю60
 61
А62
▓
 Кlayer_regularization_losses
(regularization_losses
Лnon_trainable_variables
)trainable_variables
Мlayer_metrics
*	variables
Нmetrics
Оlayers
 
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
▓
 Пlayer_regularization_losses
Рnon_trainable_variables
0trainable_variables
1regularization_losses
Сlayer_metrics
2	variables
Тmetrics
Уlayers
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
72
83
▓
 Фlayer_regularization_losses
Хnon_trainable_variables
9trainable_variables
:regularization_losses
Цlayer_metrics
;	variables
Чmetrics
Шlayers
 
 
 
▓
 Щlayer_regularization_losses
Ъnon_trainable_variables
=trainable_variables
>regularization_losses
Ыlayer_metrics
?	variables
Ьmetrics
Эlayers
 
 
 
▓
 Юlayer_regularization_losses
Яnon_trainable_variables
Atrainable_variables
Bregularization_losses
аlayer_metrics
C	variables
бmetrics
вlayers
zx
VARIABLE_VALUE$separable_conv2d_14/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_14/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
G2
 

E0
F1
G2
▓
 гlayer_regularization_losses
дnon_trainable_variables
Htrainable_variables
Iregularization_losses
еlayer_metrics
J	variables
жmetrics
зlayers
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
O2
P3
▓
 иlayer_regularization_losses
йnon_trainable_variables
Qtrainable_variables
Rregularization_losses
кlayer_metrics
S	variables
лmetrics
мlayers
 
 
 
▓
 нlayer_regularization_losses
оnon_trainable_variables
Utrainable_variables
Vregularization_losses
пlayer_metrics
W	variables
░metrics
▒layers
zx
VARIABLE_VALUE$separable_conv2d_15/depthwise_kernel@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_15/pointwise_kernel@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_15/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
[2
 

Y0
Z1
[2
▓
 ▓layer_regularization_losses
│non_trainable_variables
\trainable_variables
]regularization_losses
┤layer_metrics
^	variables
╡metrics
╢layers
 
ge
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

a0
b1
 

a0
b1
c2
d3
▓
 ╖layer_regularization_losses
╕non_trainable_variables
etrainable_variables
fregularization_losses
╣layer_metrics
g	variables
║metrics
╗layers
 
 
 
▓
 ╝layer_regularization_losses
╜non_trainable_variables
itrainable_variables
jregularization_losses
╛layer_metrics
k	variables
┐metrics
└layers
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1
 

m0
n1
▓
 ┴layer_regularization_losses
┬non_trainable_variables
otrainable_variables
pregularization_losses
├layer_metrics
q	variables
─metrics
┼layers
 
 
 
▓
 ╞layer_regularization_losses
╟non_trainable_variables
strainable_variables
tregularization_losses
╚layer_metrics
u	variables
╔metrics
╩layers
 
 
 
▓
 ╦layer_regularization_losses
╠non_trainable_variables
wtrainable_variables
xregularization_losses
═layer_metrics
y	variables
╬metrics
╧layers
zx
VARIABLE_VALUE$separable_conv2d_16/depthwise_kernel@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_16/pointwise_kernel@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_16/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
}2
 

{0
|1
}2
│
 ╨layer_regularization_losses
╤non_trainable_variables
~trainable_variables
regularization_losses
╥layer_metrics
А	variables
╙metrics
╘layers
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Г0
Д1
 
 
Г0
Д1
Е2
Ж3
╡
 ╒layer_regularization_losses
╓non_trainable_variables
Зtrainable_variables
Иregularization_losses
╫layer_metrics
Й	variables
╪metrics
┘layers
 
 
 
╡
 ┌layer_regularization_losses
█non_trainable_variables
Лtrainable_variables
Мregularization_losses
▄layer_metrics
Н	variables
▌metrics
▐layers
zx
VARIABLE_VALUE$separable_conv2d_17/depthwise_kernel@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_17/pointwise_kernel@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_17/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

П0
Р1
С2
 

П0
Р1
С2
╡
 ▀layer_regularization_losses
рnon_trainable_variables
Тtrainable_variables
Уregularization_losses
сlayer_metrics
Ф	variables
тmetrics
уlayers
 
hf
VARIABLE_VALUEbatch_normalization_20/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_20/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_20/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_20/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Ч0
Ш1
 
 
Ч0
Ш1
Щ2
Ъ3
╡
 фlayer_regularization_losses
хnon_trainable_variables
Ыtrainable_variables
Ьregularization_losses
цlayer_metrics
Э	variables
чmetrics
шlayers
 
 
 
╡
 щlayer_regularization_losses
ъnon_trainable_variables
Яtrainable_variables
аregularization_losses
ыlayer_metrics
б	variables
ьmetrics
эlayers
][
VARIABLE_VALUEconv2d_10/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_10/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

г0
д1
 

г0
д1
╡
 юlayer_regularization_losses
яnon_trainable_variables
еtrainable_variables
жregularization_losses
Ёlayer_metrics
з	variables
ёmetrics
Єlayers
 
 
 
╡
 єlayer_regularization_losses
Їnon_trainable_variables
йtrainable_variables
кregularization_losses
їlayer_metrics
л	variables
Ўmetrics
ўlayers
 
 
 
╡
 °layer_regularization_losses
∙non_trainable_variables
нtrainable_variables
оregularization_losses
·layer_metrics
п	variables
√metrics
№layers
{y
VARIABLE_VALUE$separable_conv2d_18/depthwise_kernelAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_18/pointwise_kernelAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_18/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

▒0
▓1
│2
 

▒0
▓1
│2
╡
 ¤layer_regularization_losses
■non_trainable_variables
┤trainable_variables
╡regularization_losses
 layer_metrics
╢	variables
Аmetrics
Бlayers
 
hf
VARIABLE_VALUEbatch_normalization_21/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_21/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_21/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_21/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

╣0
║1
 
 
╣0
║1
╗2
╝3
╡
 Вlayer_regularization_losses
Гnon_trainable_variables
╜trainable_variables
╛regularization_losses
Дlayer_metrics
┐	variables
Еmetrics
Жlayers
 
 
 
╡
 Зlayer_regularization_losses
Иnon_trainable_variables
┴trainable_variables
┬regularization_losses
Йlayer_metrics
├	variables
Кmetrics
Лlayers
{y
VARIABLE_VALUE$separable_conv2d_19/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_19/pointwise_kernelAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_19/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

┼0
╞1
╟2
 

┼0
╞1
╟2
╡
 Мlayer_regularization_losses
Нnon_trainable_variables
╚trainable_variables
╔regularization_losses
Оlayer_metrics
╩	variables
Пmetrics
Рlayers
 
hf
VARIABLE_VALUEbatch_normalization_22/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_22/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_22/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_22/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

═0
╬1
 
 
═0
╬1
╧2
╨3
╡
 Сlayer_regularization_losses
Тnon_trainable_variables
╤trainable_variables
╥regularization_losses
Уlayer_metrics
╙	variables
Фmetrics
Хlayers
 
 
 
╡
 Цlayer_regularization_losses
Чnon_trainable_variables
╒trainable_variables
╓regularization_losses
Шlayer_metrics
╫	variables
Щmetrics
Ъlayers
][
VARIABLE_VALUEconv2d_11/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_11/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

┘0
┌1
 

┘0
┌1
╡
 Ыlayer_regularization_losses
Ьnon_trainable_variables
█trainable_variables
▄regularization_losses
Эlayer_metrics
▌	variables
Юmetrics
Яlayers
 
 
 
╡
 аlayer_regularization_losses
бnon_trainable_variables
▀trainable_variables
рregularization_losses
вlayer_metrics
с	variables
гmetrics
дlayers
{y
VARIABLE_VALUE$separable_conv2d_20/depthwise_kernelAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_20/pointwise_kernelAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_20/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

у0
ф1
х2
 

у0
ф1
х2
╡
 еlayer_regularization_losses
жnon_trainable_variables
цtrainable_variables
чregularization_losses
зlayer_metrics
ш	variables
иmetrics
йlayers
 
hf
VARIABLE_VALUEbatch_normalization_23/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_23/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_23/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_23/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

ы0
ь1
 
 
ы0
ь1
э2
ю3
╡
 кlayer_regularization_losses
лnon_trainable_variables
яtrainable_variables
Ёregularization_losses
мlayer_metrics
ё	variables
нmetrics
оlayers
 
 
 
╡
 пlayer_regularization_losses
░non_trainable_variables
єtrainable_variables
Їregularization_losses
▒layer_metrics
ї	variables
▓metrics
│layers
 
 
 
╡
 ┤layer_regularization_losses
╡non_trainable_variables
ўtrainable_variables
°regularization_losses
╢layer_metrics
∙	variables
╖metrics
╕layers
 
 
 
╡
 ╣layer_regularization_losses
║non_trainable_variables
√trainable_variables
№regularization_losses
╗layer_metrics
¤	variables
╝metrics
╜layers
[Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_2/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
А1
 

 0
А1
╡
 ╛layer_regularization_losses
┐non_trainable_variables
Бtrainable_variables
Вregularization_losses
└layer_metrics
Г	variables
┴metrics
┬layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
А
70
81
O2
P3
c4
d5
Е6
Ж7
Щ8
Ъ9
╗10
╝11
╧12
╨13
э14
ю15
 

├0
─1
ж
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
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
 
 
 
 
 
 

70
81
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

O0
P1
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

c0
d1
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

Е0
Ж1
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

Щ0
Ъ1
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

╗0
╝1
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

╧0
╨1
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

э0
ю1
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
8

┼total

╞count
╟	variables
╚	keras_api
I

╔total

╩count
╦
_fn_kwargs
╠	variables
═	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

┼0
╞1

╟	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

╔0
╩1

╠	variables
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_14/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_14/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_15/depthwise_kernel/m\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_15/pointwise_kernel/m\layer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_15/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_16/depthwise_kernel/m\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_16/pointwise_kernel/m\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_16/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_19/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_17/depthwise_kernel/m\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_17/pointwise_kernel/m\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_17/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_20/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_10/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_10/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_18/depthwise_kernel/m]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_18/pointwise_kernel/m]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEAdam/separable_conv2d_18/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_19/depthwise_kernel/m]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_19/pointwise_kernel/m]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEAdam/separable_conv2d_19/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_22/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_11/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_11/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_20/depthwise_kernel/m]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_20/pointwise_kernel/m]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEAdam/separable_conv2d_20/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_23/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_2/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_2/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_14/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_14/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_15/depthwise_kernel/v\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_15/pointwise_kernel/v\layer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_15/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_16/depthwise_kernel/v\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_16/pointwise_kernel/v\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_16/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_19/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_17/depthwise_kernel/v\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE+Adam/separable_conv2d_17/pointwise_kernel/v\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/separable_conv2d_17/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_20/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_10/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_10/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_18/depthwise_kernel/v]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_18/pointwise_kernel/v]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEAdam/separable_conv2d_18/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_19/depthwise_kernel/v]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_19/pointwise_kernel/v]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEAdam/separable_conv2d_19/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_22/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_11/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_11/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_20/depthwise_kernel/v]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE+Adam/separable_conv2d_20/pointwise_kernel/v]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEAdam/separable_conv2d_20/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_23/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_2/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_2/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_3Placeholder*/
_output_shapes
:         @@*
dtype0*$
shape:         @@
╫
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_8/kernelconv2d_8/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance$separable_conv2d_14/depthwise_kernel$separable_conv2d_14/pointwise_kernelseparable_conv2d_14/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance$separable_conv2d_15/depthwise_kernel$separable_conv2d_15/pointwise_kernelseparable_conv2d_15/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_9/kernelconv2d_9/bias$separable_conv2d_16/depthwise_kernel$separable_conv2d_16/pointwise_kernelseparable_conv2d_16/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variance$separable_conv2d_17/depthwise_kernel$separable_conv2d_17/pointwise_kernelseparable_conv2d_17/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_10/kernelconv2d_10/bias$separable_conv2d_18/depthwise_kernel$separable_conv2d_18/pointwise_kernelseparable_conv2d_18/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variance$separable_conv2d_19/depthwise_kernel$separable_conv2d_19/pointwise_kernelseparable_conv2d_19/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_11/kernelconv2d_11/bias$separable_conv2d_20/depthwise_kernel$separable_conv2d_20/pointwise_kernelseparable_conv2d_20/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_2/kerneldense_2/bias*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_434278
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
уG
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp8separable_conv2d_14/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_14/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_14/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp8separable_conv2d_15/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_15/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_15/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp8separable_conv2d_16/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_16/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_16/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp8separable_conv2d_17/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_17/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_17/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp8separable_conv2d_18/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_18/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_18/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp8separable_conv2d_19/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_19/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_19/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp8separable_conv2d_20/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_20/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_20/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp?Adam/separable_conv2d_14/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_14/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_14/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp?Adam/separable_conv2d_15/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_15/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_15/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp?Adam/separable_conv2d_16/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_16/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_16/bias/m/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_19/beta/m/Read/ReadVariableOp?Adam/separable_conv2d_17/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_17/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_20/beta/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp?Adam/separable_conv2d_18/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_18/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp?Adam/separable_conv2d_19/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_19/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_19/bias/m/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_22/beta/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp?Adam/separable_conv2d_20/depthwise_kernel/m/Read/ReadVariableOp?Adam/separable_conv2d_20/pointwise_kernel/m/Read/ReadVariableOp3Adam/separable_conv2d_20/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp?Adam/separable_conv2d_14/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_14/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_14/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp?Adam/separable_conv2d_15/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_15/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_15/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp?Adam/separable_conv2d_16/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_16/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_16/bias/v/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_19/beta/v/Read/ReadVariableOp?Adam/separable_conv2d_17/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_17/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_20/beta/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp?Adam/separable_conv2d_18/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_18/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp?Adam/separable_conv2d_19/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_19/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_19/bias/v/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_22/beta/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp?Adam/separable_conv2d_20/depthwise_kernel/v/Read/ReadVariableOp?Adam/separable_conv2d_20/pointwise_kernel/v/Read/ReadVariableOp3Adam/separable_conv2d_20/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*╢
Tinо
л2и	*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_436789
ц-
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance$separable_conv2d_14/depthwise_kernel$separable_conv2d_14/pointwise_kernelseparable_conv2d_14/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance$separable_conv2d_15/depthwise_kernel$separable_conv2d_15/pointwise_kernelseparable_conv2d_15/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_9/kernelconv2d_9/bias$separable_conv2d_16/depthwise_kernel$separable_conv2d_16/pointwise_kernelseparable_conv2d_16/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variance$separable_conv2d_17/depthwise_kernel$separable_conv2d_17/pointwise_kernelseparable_conv2d_17/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_10/kernelconv2d_10/bias$separable_conv2d_18/depthwise_kernel$separable_conv2d_18/pointwise_kernelseparable_conv2d_18/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variance$separable_conv2d_19/depthwise_kernel$separable_conv2d_19/pointwise_kernelseparable_conv2d_19/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_11/kernelconv2d_11/bias$separable_conv2d_20/depthwise_kernel$separable_conv2d_20/pointwise_kernelseparable_conv2d_20/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_8/kernel/mAdam/conv2d_8/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/m+Adam/separable_conv2d_14/depthwise_kernel/m+Adam/separable_conv2d_14/pointwise_kernel/mAdam/separable_conv2d_14/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/m+Adam/separable_conv2d_15/depthwise_kernel/m+Adam/separable_conv2d_15/pointwise_kernel/mAdam/separable_conv2d_15/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/m+Adam/separable_conv2d_16/depthwise_kernel/m+Adam/separable_conv2d_16/pointwise_kernel/mAdam/separable_conv2d_16/bias/m#Adam/batch_normalization_19/gamma/m"Adam/batch_normalization_19/beta/m+Adam/separable_conv2d_17/depthwise_kernel/m+Adam/separable_conv2d_17/pointwise_kernel/mAdam/separable_conv2d_17/bias/m#Adam/batch_normalization_20/gamma/m"Adam/batch_normalization_20/beta/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/m+Adam/separable_conv2d_18/depthwise_kernel/m+Adam/separable_conv2d_18/pointwise_kernel/mAdam/separable_conv2d_18/bias/m#Adam/batch_normalization_21/gamma/m"Adam/batch_normalization_21/beta/m+Adam/separable_conv2d_19/depthwise_kernel/m+Adam/separable_conv2d_19/pointwise_kernel/mAdam/separable_conv2d_19/bias/m#Adam/batch_normalization_22/gamma/m"Adam/batch_normalization_22/beta/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/m+Adam/separable_conv2d_20/depthwise_kernel/m+Adam/separable_conv2d_20/pointwise_kernel/mAdam/separable_conv2d_20/bias/m#Adam/batch_normalization_23/gamma/m"Adam/batch_normalization_23/beta/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/v+Adam/separable_conv2d_14/depthwise_kernel/v+Adam/separable_conv2d_14/pointwise_kernel/vAdam/separable_conv2d_14/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/v+Adam/separable_conv2d_15/depthwise_kernel/v+Adam/separable_conv2d_15/pointwise_kernel/vAdam/separable_conv2d_15/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/v+Adam/separable_conv2d_16/depthwise_kernel/v+Adam/separable_conv2d_16/pointwise_kernel/vAdam/separable_conv2d_16/bias/v#Adam/batch_normalization_19/gamma/v"Adam/batch_normalization_19/beta/v+Adam/separable_conv2d_17/depthwise_kernel/v+Adam/separable_conv2d_17/pointwise_kernel/vAdam/separable_conv2d_17/bias/v#Adam/batch_normalization_20/gamma/v"Adam/batch_normalization_20/beta/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/v+Adam/separable_conv2d_18/depthwise_kernel/v+Adam/separable_conv2d_18/pointwise_kernel/vAdam/separable_conv2d_18/bias/v#Adam/batch_normalization_21/gamma/v"Adam/batch_normalization_21/beta/v+Adam/separable_conv2d_19/depthwise_kernel/v+Adam/separable_conv2d_19/pointwise_kernel/vAdam/separable_conv2d_19/bias/v#Adam/batch_normalization_22/gamma/v"Adam/batch_normalization_22/beta/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/v+Adam/separable_conv2d_20/depthwise_kernel/v+Adam/separable_conv2d_20/pointwise_kernel/vAdam/separable_conv2d_20/bias/v#Adam/batch_normalization_23/gamma/v"Adam/batch_normalization_23/beta/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*╡
Tinн
к2з*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_437297д╩)
Е
e
I__inference_activation_20_layer_call_and_return_conditional_losses_432378

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_431847

inputsC
(separable_conv2d_readvariableop_resource:╪F
*separable_conv2d_readvariableop_1_resource:╪╪.
biasadd_readvariableop_resource:	╪
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪╪*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ╪*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           ╪*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ╪2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ╪: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
┌
L
0__inference_max_pooling2d_6_layer_call_fn_431357

inputs
identityя
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
GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_4313512
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ў
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_432593

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╗

А
D__inference_conv2d_9_layer_call_and_return_conditional_losses_435468

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_433159

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_430977

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
э
J
.__inference_activation_22_layer_call_fn_435932

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_4324852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╪:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_432875

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
Е
e
I__inference_activation_22_layer_call_and_return_conditional_losses_432485

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╪2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╪:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_16_layer_call_fn_435154

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4309772
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╗
W
;__inference_global_average_pooling2d_2_layer_call_fn_432164

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_4321582
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_432013

inputsC
(separable_conv2d_readvariableop_resource:╪F
*separable_conv2d_readvariableop_1_resource:╪А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪А*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ╪*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ╪: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
╖

ї
C__inference_dense_2_layer_call_and_return_conditional_losses_436259

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
█
$__inference_signature_wrapper_434278
input_3"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:А&

unknown_13:АА

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А&

unknown_19:АА

unknown_20:	А%

unknown_21:А&

unknown_22:АА

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А%

unknown_28:А&

unknown_29:АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А&

unknown_35:АА

unknown_36:	А%

unknown_37:А&

unknown_38:А╪

unknown_39:	╪

unknown_40:	╪

unknown_41:	╪

unknown_42:	╪

unknown_43:	╪%

unknown_44:╪&

unknown_45:╪╪

unknown_46:	╪

unknown_47:	╪

unknown_48:	╪

unknown_49:	╪

unknown_50:	╪&

unknown_51:А╪

unknown_52:	╪%

unknown_53:╪&

unknown_54:╪А

unknown_55:	А

unknown_56:	А

unknown_57:	А

unknown_58:	А

unknown_59:	А

unknown_60:	А


unknown_61:

identityИвStatefulPartitionedCallв	
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_4309112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         @@
!
_user_specified_name	input_3
╘
б
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435553

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
э
J
.__inference_activation_21_layer_call_fn_435798

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_4324442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_432091

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435852

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
Э├
╩
C__inference_model_2_layer_call_and_return_conditional_losses_432613

inputs*
conv2d_8_432186:А
conv2d_8_432188:	А,
batch_normalization_16_432209:	А,
batch_normalization_16_432211:	А,
batch_normalization_16_432213:	А,
batch_normalization_16_432215:	А5
separable_conv2d_14_432232:А6
separable_conv2d_14_432234:АА)
separable_conv2d_14_432236:	А,
batch_normalization_17_432257:	А,
batch_normalization_17_432259:	А,
batch_normalization_17_432261:	А,
batch_normalization_17_432263:	А5
separable_conv2d_15_432273:А6
separable_conv2d_15_432275:АА)
separable_conv2d_15_432277:	А,
batch_normalization_18_432298:	А,
batch_normalization_18_432300:	А,
batch_normalization_18_432302:	А,
batch_normalization_18_432304:	А+
conv2d_9_432319:АА
conv2d_9_432321:	А5
separable_conv2d_16_432339:А6
separable_conv2d_16_432341:АА)
separable_conv2d_16_432343:	А,
batch_normalization_19_432364:	А,
batch_normalization_19_432366:	А,
batch_normalization_19_432368:	А,
batch_normalization_19_432370:	А5
separable_conv2d_17_432380:А6
separable_conv2d_17_432382:АА)
separable_conv2d_17_432384:	А,
batch_normalization_20_432405:	А,
batch_normalization_20_432407:	А,
batch_normalization_20_432409:	А,
batch_normalization_20_432411:	А,
conv2d_10_432426:АА
conv2d_10_432428:	А5
separable_conv2d_18_432446:А6
separable_conv2d_18_432448:А╪)
separable_conv2d_18_432450:	╪,
batch_normalization_21_432471:	╪,
batch_normalization_21_432473:	╪,
batch_normalization_21_432475:	╪,
batch_normalization_21_432477:	╪5
separable_conv2d_19_432487:╪6
separable_conv2d_19_432489:╪╪)
separable_conv2d_19_432491:	╪,
batch_normalization_22_432512:	╪,
batch_normalization_22_432514:	╪,
batch_normalization_22_432516:	╪,
batch_normalization_22_432518:	╪,
conv2d_11_432533:А╪
conv2d_11_432535:	╪5
separable_conv2d_20_432546:╪6
separable_conv2d_20_432548:╪А)
separable_conv2d_20_432550:	А,
batch_normalization_23_432571:	А,
batch_normalization_23_432573:	А,
batch_normalization_23_432575:	А,
batch_normalization_23_432577:	А!
dense_2_432607:	А

dense_2_432609:

identityИв.batch_normalization_16/StatefulPartitionedCallв.batch_normalization_17/StatefulPartitionedCallв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв+separable_conv2d_16/StatefulPartitionedCallв+separable_conv2d_17/StatefulPartitionedCallв+separable_conv2d_18/StatefulPartitionedCallв+separable_conv2d_19/StatefulPartitionedCallв+separable_conv2d_20/StatefulPartitionedCallm
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_2/Cast/xq
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_2/Cast_1/xИ
rescaling_2/mulMulinputsrescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/mulЩ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/addн
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallrescaling_2/add:z:0conv2d_8_432186conv2d_8_432188*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_4321852"
 conv2d_8/StatefulPartitionedCall╦
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_16_432209batch_normalization_16_432211batch_normalization_16_432213batch_normalization_16_432215*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_43220820
.batch_normalization_16/StatefulPartitionedCallа
activation_16/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_4322232
activation_16/PartitionedCallП
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_4322302
activation_17/PartitionedCallХ
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0separable_conv2d_14_432232separable_conv2d_14_432234separable_conv2d_14_432236*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_4310532-
+separable_conv2d_14/StatefulPartitionedCall╓
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0batch_normalization_17_432257batch_normalization_17_432259batch_normalization_17_432261batch_normalization_17_432263*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_43225620
.batch_normalization_17/StatefulPartitionedCallа
activation_18/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_4322712
activation_18/PartitionedCallХ
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0separable_conv2d_15_432273separable_conv2d_15_432275separable_conv2d_15_432277*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_4312072-
+separable_conv2d_15/StatefulPartitionedCall╓
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_432298batch_normalization_18_432300batch_normalization_18_432302batch_normalization_18_432304*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_43229720
.batch_normalization_18/StatefulPartitionedCallж
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_4313512!
max_pooling2d_6/PartitionedCall└
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_9_432319conv2d_9_432321*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_4323182"
 conv2d_9/StatefulPartitionedCallе
add_6/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_4323302
add_6/PartitionedCallЗ
activation_19/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_4323372
activation_19/PartitionedCallХ
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0separable_conv2d_16_432339separable_conv2d_16_432341separable_conv2d_16_432343*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_4313732-
+separable_conv2d_16/StatefulPartitionedCall╓
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_432364batch_normalization_19_432366batch_normalization_19_432368batch_normalization_19_432370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_43236320
.batch_normalization_19/StatefulPartitionedCallа
activation_20/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_4323782
activation_20/PartitionedCallХ
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0separable_conv2d_17_432380separable_conv2d_17_432382separable_conv2d_17_432384*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_4315272-
+separable_conv2d_17/StatefulPartitionedCall╓
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_432405batch_normalization_20_432407batch_normalization_20_432409batch_normalization_20_432411*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_43240420
.batch_normalization_20/StatefulPartitionedCallж
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_4316712!
max_pooling2d_7/PartitionedCall╜
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_10_432426conv2d_10_432428*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_4324252#
!conv2d_10/StatefulPartitionedCallж
add_7/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_4324372
add_7/PartitionedCallЗ
activation_21/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_4324442
activation_21/PartitionedCallХ
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0separable_conv2d_18_432446separable_conv2d_18_432448separable_conv2d_18_432450*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_4316932-
+separable_conv2d_18/StatefulPartitionedCall╓
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_432471batch_normalization_21_432473batch_normalization_21_432475batch_normalization_21_432477*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_43247020
.batch_normalization_21/StatefulPartitionedCallа
activation_22/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_4324852
activation_22/PartitionedCallХ
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0separable_conv2d_19_432487separable_conv2d_19_432489separable_conv2d_19_432491*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_4318472-
+separable_conv2d_19/StatefulPartitionedCall╓
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_432512batch_normalization_22_432514batch_normalization_22_432516batch_normalization_22_432518*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_43251120
.batch_normalization_22/StatefulPartitionedCallж
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_4319912!
max_pooling2d_8/PartitionedCall╜
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCalladd_7/PartitionedCall:output:0conv2d_11_432533conv2d_11_432535*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_4325322#
!conv2d_11/StatefulPartitionedCallж
add_8/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_8_layer_call_and_return_conditional_losses_4325442
add_8/PartitionedCallН
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCalladd_8/PartitionedCall:output:0separable_conv2d_20_432546separable_conv2d_20_432548separable_conv2d_20_432550*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_4320132-
+separable_conv2d_20/StatefulPartitionedCall╓
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_432571batch_normalization_23_432573batch_normalization_23_432575batch_normalization_23_432577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_43257020
.batch_normalization_23/StatefulPartitionedCallа
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_4325852
activation_23/PartitionedCallо
*global_average_pooling2d_2/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_4321582,
*global_average_pooling2d_2/PartitionedCallИ
dropout_2/PartitionedCallPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4325932
dropout_2/PartitionedCallо
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_432607dense_2_432609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4326062!
dense_2/StatefulPartitionedCallЎ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_432925

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_16_layer_call_fn_435167

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4322082
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_16_layer_call_fn_435180

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4332152
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_432363

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Е
e
I__inference_activation_16_layer_call_and_return_conditional_losses_432223

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:           А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_436004

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
р
╥
4__inference_separable_conv2d_17_layer_call_fn_431539

inputs"
unknown:А%
	unknown_0:АА
	unknown_1:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_4315272
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435092

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435236

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╝

Б
E__inference_conv2d_10_layer_call_and_return_conditional_losses_435767

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_431727

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
┤
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_432772

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_21_layer_call_fn_435883

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4317272
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_19_layer_call_fn_435610

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4323632
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_23_layer_call_fn_436211

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4328142
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435986

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_18_layer_call_fn_435445

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4322972
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_432814

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436105

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Є├
№A
C__inference_model_2_layer_call_and_return_conditional_losses_434775

inputsB
'conv2d_8_conv2d_readvariableop_resource:А7
(conv2d_8_biasadd_readvariableop_resource:	А=
.batch_normalization_16_readvariableop_resource:	А?
0batch_normalization_16_readvariableop_1_resource:	АN
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	АW
<separable_conv2d_14_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_14_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_14_biasadd_readvariableop_resource:	А=
.batch_normalization_17_readvariableop_resource:	А?
0batch_normalization_17_readvariableop_1_resource:	АN
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	АW
<separable_conv2d_15_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_15_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_15_biasadd_readvariableop_resource:	А=
.batch_normalization_18_readvariableop_resource:	А?
0batch_normalization_18_readvariableop_1_resource:	АN
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	АC
'conv2d_9_conv2d_readvariableop_resource:АА7
(conv2d_9_biasadd_readvariableop_resource:	АW
<separable_conv2d_16_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_16_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_16_biasadd_readvariableop_resource:	А=
.batch_normalization_19_readvariableop_resource:	А?
0batch_normalization_19_readvariableop_1_resource:	АN
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	АW
<separable_conv2d_17_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_17_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_17_biasadd_readvariableop_resource:	А=
.batch_normalization_20_readvariableop_resource:	А?
0batch_normalization_20_readvariableop_1_resource:	АN
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:	АD
(conv2d_10_conv2d_readvariableop_resource:АА8
)conv2d_10_biasadd_readvariableop_resource:	АW
<separable_conv2d_18_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_18_separable_conv2d_readvariableop_1_resource:А╪B
3separable_conv2d_18_biasadd_readvariableop_resource:	╪=
.batch_normalization_21_readvariableop_resource:	╪?
0batch_normalization_21_readvariableop_1_resource:	╪N
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	╪P
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	╪W
<separable_conv2d_19_separable_conv2d_readvariableop_resource:╪Z
>separable_conv2d_19_separable_conv2d_readvariableop_1_resource:╪╪B
3separable_conv2d_19_biasadd_readvariableop_resource:	╪=
.batch_normalization_22_readvariableop_resource:	╪?
0batch_normalization_22_readvariableop_1_resource:	╪N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	╪P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	╪D
(conv2d_11_conv2d_readvariableop_resource:А╪8
)conv2d_11_biasadd_readvariableop_resource:	╪W
<separable_conv2d_20_separable_conv2d_readvariableop_resource:╪Z
>separable_conv2d_20_separable_conv2d_readvariableop_1_resource:╪АB
3separable_conv2d_20_biasadd_readvariableop_resource:	А=
.batch_normalization_23_readvariableop_resource:	А?
0batch_normalization_23_readvariableop_1_resource:	АN
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	А9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:

identityИв%batch_normalization_16/AssignNewValueв'batch_normalization_16/AssignNewValue_1в6batch_normalization_16/FusedBatchNormV3/ReadVariableOpв8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_16/ReadVariableOpв'batch_normalization_16/ReadVariableOp_1в%batch_normalization_17/AssignNewValueв'batch_normalization_17/AssignNewValue_1в6batch_normalization_17/FusedBatchNormV3/ReadVariableOpв8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_17/ReadVariableOpв'batch_normalization_17/ReadVariableOp_1в%batch_normalization_18/AssignNewValueв'batch_normalization_18/AssignNewValue_1в6batch_normalization_18/FusedBatchNormV3/ReadVariableOpв8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_18/ReadVariableOpв'batch_normalization_18/ReadVariableOp_1в%batch_normalization_19/AssignNewValueв'batch_normalization_19/AssignNewValue_1в6batch_normalization_19/FusedBatchNormV3/ReadVariableOpв8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_19/ReadVariableOpв'batch_normalization_19/ReadVariableOp_1в%batch_normalization_20/AssignNewValueв'batch_normalization_20/AssignNewValue_1в6batch_normalization_20/FusedBatchNormV3/ReadVariableOpв8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_20/ReadVariableOpв'batch_normalization_20/ReadVariableOp_1в%batch_normalization_21/AssignNewValueв'batch_normalization_21/AssignNewValue_1в6batch_normalization_21/FusedBatchNormV3/ReadVariableOpв8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_21/ReadVariableOpв'batch_normalization_21/ReadVariableOp_1в%batch_normalization_22/AssignNewValueв'batch_normalization_22/AssignNewValue_1в6batch_normalization_22/FusedBatchNormV3/ReadVariableOpв8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_22/ReadVariableOpв'batch_normalization_22/ReadVariableOp_1в%batch_normalization_23/AssignNewValueв'batch_normalization_23/AssignNewValue_1в6batch_normalization_23/FusedBatchNormV3/ReadVariableOpв8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_23/ReadVariableOpв'batch_normalization_23/ReadVariableOp_1в conv2d_10/BiasAdd/ReadVariableOpвconv2d_10/Conv2D/ReadVariableOpв conv2d_11/BiasAdd/ReadVariableOpвconv2d_11/Conv2D/ReadVariableOpвconv2d_8/BiasAdd/ReadVariableOpвconv2d_8/Conv2D/ReadVariableOpвconv2d_9/BiasAdd/ReadVariableOpвconv2d_9/Conv2D/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpв*separable_conv2d_14/BiasAdd/ReadVariableOpв3separable_conv2d_14/separable_conv2d/ReadVariableOpв5separable_conv2d_14/separable_conv2d/ReadVariableOp_1в*separable_conv2d_15/BiasAdd/ReadVariableOpв3separable_conv2d_15/separable_conv2d/ReadVariableOpв5separable_conv2d_15/separable_conv2d/ReadVariableOp_1в*separable_conv2d_16/BiasAdd/ReadVariableOpв3separable_conv2d_16/separable_conv2d/ReadVariableOpв5separable_conv2d_16/separable_conv2d/ReadVariableOp_1в*separable_conv2d_17/BiasAdd/ReadVariableOpв3separable_conv2d_17/separable_conv2d/ReadVariableOpв5separable_conv2d_17/separable_conv2d/ReadVariableOp_1в*separable_conv2d_18/BiasAdd/ReadVariableOpв3separable_conv2d_18/separable_conv2d/ReadVariableOpв5separable_conv2d_18/separable_conv2d/ReadVariableOp_1в*separable_conv2d_19/BiasAdd/ReadVariableOpв3separable_conv2d_19/separable_conv2d/ReadVariableOpв5separable_conv2d_19/separable_conv2d/ReadVariableOp_1в*separable_conv2d_20/BiasAdd/ReadVariableOpв3separable_conv2d_20/separable_conv2d/ReadVariableOpв5separable_conv2d_20/separable_conv2d/ReadVariableOp_1m
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_2/Cast/xq
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_2/Cast_1/xИ
rescaling_2/mulMulinputsrescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/mulЩ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/add▒
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02 
conv2d_8/Conv2D/ReadVariableOp╠
conv2d_8/Conv2DConv2Drescaling_2/add:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
conv2d_8/Conv2Dи
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpн
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
conv2d_8/BiasAdd║
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_16/ReadVariableOp└
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_16/ReadVariableOp_1э
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1·
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_16/FusedBatchNormV3╡
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue┴
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1Ш
activation_16/ReluRelu+batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:           А2
activation_16/ReluН
activation_17/ReluRelu activation_16/Relu:activations:0*
T0*0
_output_shapes
:           А2
activation_17/ReluЁ
3separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_14_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_14/separable_conv2d/ReadVariableOpў
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_14_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2,
*separable_conv2d_14/separable_conv2d/Shape╣
2separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_14/separable_conv2d/dilation_rate╗
.separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNative activation_17/Relu:activations:0;separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_14/separable_conv2d/depthwise▓
$separable_conv2d_14/separable_conv2dConv2D7separable_conv2d_14/separable_conv2d/depthwise:output:0=separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_14/separable_conv2d╔
*separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_14/BiasAdd/ReadVariableOpу
separable_conv2d_14/BiasAddBiasAdd-separable_conv2d_14/separable_conv2d:output:02separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_14/BiasAdd║
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_17/ReadVariableOp└
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_17/ReadVariableOp_1э
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_14/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_17/FusedBatchNormV3╡
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue┴
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1Ш
activation_18/ReluRelu+batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:           А2
activation_18/ReluЁ
3separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_15_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_15/separable_conv2d/ReadVariableOpў
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_15_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_15/separable_conv2d/Shape╣
2separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_15/separable_conv2d/dilation_rate╗
.separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative activation_18/Relu:activations:0;separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_15/separable_conv2d/depthwise▓
$separable_conv2d_15/separable_conv2dConv2D7separable_conv2d_15/separable_conv2d/depthwise:output:0=separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_15/separable_conv2d╔
*separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_15/BiasAdd/ReadVariableOpу
separable_conv2d_15/BiasAddBiasAdd-separable_conv2d_15/separable_conv2d:output:02separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_15/BiasAdd║
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_18/ReadVariableOp└
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_18/ReadVariableOp_1э
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_15/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_18/FusedBatchNormV3╡
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_18/AssignNewValue┴
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_18/AssignNewValue_1╫
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_18/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d_6/MaxPool▓
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_9/Conv2D/ReadVariableOp┘
conv2d_9/Conv2DConv2D activation_16/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_9/Conv2Dи
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_9/BiasAdd/ReadVariableOpн
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_9/BiasAddЧ
	add_6/addAddV2 max_pooling2d_6/MaxPool:output:0conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
	add_6/addz
activation_19/ReluReluadd_6/add:z:0*
T0*0
_output_shapes
:         А2
activation_19/ReluЁ
3separable_conv2d_16/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_16_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_16/separable_conv2d/ReadVariableOpў
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_16_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_16/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_16/separable_conv2d/Shape╣
2separable_conv2d_16/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_16/separable_conv2d/dilation_rate╗
.separable_conv2d_16/separable_conv2d/depthwiseDepthwiseConv2dNative activation_19/Relu:activations:0;separable_conv2d_16/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
20
.separable_conv2d_16/separable_conv2d/depthwise▓
$separable_conv2d_16/separable_conv2dConv2D7separable_conv2d_16/separable_conv2d/depthwise:output:0=separable_conv2d_16/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2&
$separable_conv2d_16/separable_conv2d╔
*separable_conv2d_16/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_16/BiasAdd/ReadVariableOpу
separable_conv2d_16/BiasAddBiasAdd-separable_conv2d_16/separable_conv2d:output:02separable_conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
separable_conv2d_16/BiasAdd║
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_19/ReadVariableOp└
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_19/ReadVariableOp_1э
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_16/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_19/FusedBatchNormV3╡
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_19/AssignNewValue┴
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_19/AssignNewValue_1Ш
activation_20/ReluRelu+batch_normalization_19/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
activation_20/ReluЁ
3separable_conv2d_17/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_17_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_17/separable_conv2d/ReadVariableOpў
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_17_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_17/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_17/separable_conv2d/Shape╣
2separable_conv2d_17/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_17/separable_conv2d/dilation_rate╗
.separable_conv2d_17/separable_conv2d/depthwiseDepthwiseConv2dNative activation_20/Relu:activations:0;separable_conv2d_17/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
20
.separable_conv2d_17/separable_conv2d/depthwise▓
$separable_conv2d_17/separable_conv2dConv2D7separable_conv2d_17/separable_conv2d/depthwise:output:0=separable_conv2d_17/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2&
$separable_conv2d_17/separable_conv2d╔
*separable_conv2d_17/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_17/BiasAdd/ReadVariableOpу
separable_conv2d_17/BiasAddBiasAdd-separable_conv2d_17/separable_conv2d:output:02separable_conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
separable_conv2d_17/BiasAdd║
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_20/ReadVariableOp└
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_20/ReadVariableOp_1э
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_17/BiasAdd:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_20/FusedBatchNormV3╡
%batch_normalization_20/AssignNewValueAssignVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource4batch_normalization_20/FusedBatchNormV3:batch_mean:07^batch_normalization_20/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_20/AssignNewValue┴
'batch_normalization_20/AssignNewValue_1AssignVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_20/FusedBatchNormV3:batch_variance:09^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_20/AssignNewValue_1╫
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_20/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d_7/MaxPool╡
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_10/Conv2D/ReadVariableOp╔
conv2d_10/Conv2DConv2Dadd_6/add:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_10/Conv2Dл
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▒
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_10/BiasAddШ
	add_7/addAddV2 max_pooling2d_7/MaxPool:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
	add_7/addz
activation_21/ReluReluadd_7/add:z:0*
T0*0
_output_shapes
:         А2
activation_21/ReluЁ
3separable_conv2d_18/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_18_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_18/separable_conv2d/ReadVariableOpў
5separable_conv2d_18/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_18_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:А╪*
dtype027
5separable_conv2d_18/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_18/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_18/separable_conv2d/Shape╣
2separable_conv2d_18/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_18/separable_conv2d/dilation_rate╗
.separable_conv2d_18/separable_conv2d/depthwiseDepthwiseConv2dNative activation_21/Relu:activations:0;separable_conv2d_18/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
20
.separable_conv2d_18/separable_conv2d/depthwise▓
$separable_conv2d_18/separable_conv2dConv2D7separable_conv2d_18/separable_conv2d/depthwise:output:0=separable_conv2d_18/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         ╪*
paddingVALID*
strides
2&
$separable_conv2d_18/separable_conv2d╔
*separable_conv2d_18/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02,
*separable_conv2d_18/BiasAdd/ReadVariableOpу
separable_conv2d_18/BiasAddBiasAdd-separable_conv2d_18/separable_conv2d:output:02separable_conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
separable_conv2d_18/BiasAdd║
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:╪*
dtype02'
%batch_normalization_21/ReadVariableOp└
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02)
'batch_normalization_21/ReadVariableOp_1э
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_18/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_21/FusedBatchNormV3╡
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_21/AssignNewValue┴
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_21/AssignNewValue_1Ш
activation_22/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╪2
activation_22/ReluЁ
3separable_conv2d_19/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_19_separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype025
3separable_conv2d_19/separable_conv2d/ReadVariableOpў
5separable_conv2d_19/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_19_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪╪*
dtype027
5separable_conv2d_19/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_19/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     2,
*separable_conv2d_19/separable_conv2d/Shape╣
2separable_conv2d_19/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_19/separable_conv2d/dilation_rate╗
.separable_conv2d_19/separable_conv2d/depthwiseDepthwiseConv2dNative activation_22/Relu:activations:0;separable_conv2d_19/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
20
.separable_conv2d_19/separable_conv2d/depthwise▓
$separable_conv2d_19/separable_conv2dConv2D7separable_conv2d_19/separable_conv2d/depthwise:output:0=separable_conv2d_19/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         ╪*
paddingVALID*
strides
2&
$separable_conv2d_19/separable_conv2d╔
*separable_conv2d_19/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02,
*separable_conv2d_19/BiasAdd/ReadVariableOpу
separable_conv2d_19/BiasAddBiasAdd-separable_conv2d_19/separable_conv2d:output:02separable_conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
separable_conv2d_19/BiasAdd║
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:╪*
dtype02'
%batch_normalization_22/ReadVariableOp└
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02)
'batch_normalization_22/ReadVariableOp_1э
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype028
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02:
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_19/BiasAdd:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_22/FusedBatchNormV3╡
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_22/AssignNewValue┴
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_22/AssignNewValue_1╫
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_22/FusedBatchNormV3:y:0*0
_output_shapes
:         ╪*
ksize
*
paddingSAME*
strides
2
max_pooling2d_8/MaxPool╡
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:А╪*
dtype02!
conv2d_11/Conv2D/ReadVariableOp╔
conv2d_11/Conv2DConv2Dadd_7/add:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
2
conv2d_11/Conv2Dл
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp▒
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
conv2d_11/BiasAddШ
	add_8/addAddV2 max_pooling2d_8/MaxPool:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         ╪2
	add_8/addЁ
3separable_conv2d_20/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_20_separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype025
3separable_conv2d_20/separable_conv2d/ReadVariableOpў
5separable_conv2d_20/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_20_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪А*
dtype027
5separable_conv2d_20/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_20/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     2,
*separable_conv2d_20/separable_conv2d/Shape╣
2separable_conv2d_20/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_20/separable_conv2d/dilation_rateи
.separable_conv2d_20/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_8/add:z:0;separable_conv2d_20/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
20
.separable_conv2d_20/separable_conv2d/depthwise▓
$separable_conv2d_20/separable_conv2dConv2D7separable_conv2d_20/separable_conv2d/depthwise:output:0=separable_conv2d_20/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2&
$separable_conv2d_20/separable_conv2d╔
*separable_conv2d_20/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_20/BiasAdd/ReadVariableOpу
separable_conv2d_20/BiasAddBiasAdd-separable_conv2d_20/separable_conv2d:output:02separable_conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
separable_conv2d_20/BiasAdd║
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_23/ReadVariableOp└
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_23/ReadVariableOp_1э
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Е
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_20/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_23/FusedBatchNormV3╡
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_23/AssignNewValue┴
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_23/AssignNewValue_1Ш
activation_23/ReluRelu+batch_normalization_23/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
activation_23/Relu╖
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_2/Mean/reduction_indices█
global_average_pooling2d_2/MeanMean activation_23/Relu:activations:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2!
global_average_pooling2d_2/Meanw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const┤
dropout_2/dropout/MulMul(global_average_pooling2d_2/Mean:output:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_2/dropout/MulК
dropout_2/dropout/ShapeShape(global_average_pooling2d_2/Mean:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape╙
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yч
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2 
dropout_2/dropout/GreaterEqualЮ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_2/dropout/Castг
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_2/dropout/Mul_1ж
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_2/MatMul/ReadVariableOpа
dense_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_2/Softmax╗
IdentityIdentitydense_2/Softmax:softmax:0&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1&^batch_normalization_20/AssignNewValue(^batch_normalization_20/AssignNewValue_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp+^separable_conv2d_14/BiasAdd/ReadVariableOp4^separable_conv2d_14/separable_conv2d/ReadVariableOp6^separable_conv2d_14/separable_conv2d/ReadVariableOp_1+^separable_conv2d_15/BiasAdd/ReadVariableOp4^separable_conv2d_15/separable_conv2d/ReadVariableOp6^separable_conv2d_15/separable_conv2d/ReadVariableOp_1+^separable_conv2d_16/BiasAdd/ReadVariableOp4^separable_conv2d_16/separable_conv2d/ReadVariableOp6^separable_conv2d_16/separable_conv2d/ReadVariableOp_1+^separable_conv2d_17/BiasAdd/ReadVariableOp4^separable_conv2d_17/separable_conv2d/ReadVariableOp6^separable_conv2d_17/separable_conv2d/ReadVariableOp_1+^separable_conv2d_18/BiasAdd/ReadVariableOp4^separable_conv2d_18/separable_conv2d/ReadVariableOp6^separable_conv2d_18/separable_conv2d/ReadVariableOp_1+^separable_conv2d_19/BiasAdd/ReadVariableOp4^separable_conv2d_19/separable_conv2d/ReadVariableOp6^separable_conv2d_19/separable_conv2d/ReadVariableOp_1+^separable_conv2d_20/BiasAdd/ReadVariableOp4^separable_conv2d_20/separable_conv2d/ReadVariableOp6^separable_conv2d_20/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12N
%batch_normalization_20/AssignNewValue%batch_normalization_20/AssignNewValue2R
'batch_normalization_20/AssignNewValue_1'batch_normalization_20/AssignNewValue_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2X
*separable_conv2d_14/BiasAdd/ReadVariableOp*separable_conv2d_14/BiasAdd/ReadVariableOp2j
3separable_conv2d_14/separable_conv2d/ReadVariableOp3separable_conv2d_14/separable_conv2d/ReadVariableOp2n
5separable_conv2d_14/separable_conv2d/ReadVariableOp_15separable_conv2d_14/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_15/BiasAdd/ReadVariableOp*separable_conv2d_15/BiasAdd/ReadVariableOp2j
3separable_conv2d_15/separable_conv2d/ReadVariableOp3separable_conv2d_15/separable_conv2d/ReadVariableOp2n
5separable_conv2d_15/separable_conv2d/ReadVariableOp_15separable_conv2d_15/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_16/BiasAdd/ReadVariableOp*separable_conv2d_16/BiasAdd/ReadVariableOp2j
3separable_conv2d_16/separable_conv2d/ReadVariableOp3separable_conv2d_16/separable_conv2d/ReadVariableOp2n
5separable_conv2d_16/separable_conv2d/ReadVariableOp_15separable_conv2d_16/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_17/BiasAdd/ReadVariableOp*separable_conv2d_17/BiasAdd/ReadVariableOp2j
3separable_conv2d_17/separable_conv2d/ReadVariableOp3separable_conv2d_17/separable_conv2d/ReadVariableOp2n
5separable_conv2d_17/separable_conv2d/ReadVariableOp_15separable_conv2d_17/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_18/BiasAdd/ReadVariableOp*separable_conv2d_18/BiasAdd/ReadVariableOp2j
3separable_conv2d_18/separable_conv2d/ReadVariableOp3separable_conv2d_18/separable_conv2d/ReadVariableOp2n
5separable_conv2d_18/separable_conv2d/ReadVariableOp_15separable_conv2d_18/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_19/BiasAdd/ReadVariableOp*separable_conv2d_19/BiasAdd/ReadVariableOp2j
3separable_conv2d_19/separable_conv2d/ReadVariableOp3separable_conv2d_19/separable_conv2d/ReadVariableOp2n
5separable_conv2d_19/separable_conv2d/ReadVariableOp_15separable_conv2d_19/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_20/BiasAdd/ReadVariableOp*separable_conv2d_20/BiasAdd/ReadVariableOp2j
3separable_conv2d_20/separable_conv2d/ReadVariableOp3separable_conv2d_20/separable_conv2d/ReadVariableOp2n
5separable_conv2d_20/separable_conv2d/ReadVariableOp_15separable_conv2d_20/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
э
J
.__inference_activation_18_layer_call_fn_435334

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_4322712
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_431241

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
 
k
A__inference_add_7_layer_call_and_return_conditional_losses_432437

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:         А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs:XT
0
_output_shapes
:         А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_431925

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435968

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_17_layer_call_fn_435324

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4331592
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435218

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_23_layer_call_fn_436185

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4320912
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_22_layer_call_fn_436017

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4318812
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_433215

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_432511

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_21_layer_call_fn_435922

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4329252
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
Е
e
I__inference_activation_19_layer_call_and_return_conditional_losses_432337

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435834

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
э
J
.__inference_activation_16_layer_call_fn_435190

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_4322232
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
к
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_431991

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_431053

inputsC
(separable_conv2d_readvariableop_resource:АF
*separable_conv2d_readvariableop_1_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_433042

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┼
F
*__inference_dropout_2_layer_call_fn_436243

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4325932
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_431087

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Е
e
I__inference_activation_18_layer_call_and_return_conditional_losses_435329

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:           А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Е
e
I__inference_activation_17_layer_call_and_return_conditional_losses_435195

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:           А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_431527

inputsC
(separable_conv2d_readvariableop_resource:АF
*separable_conv2d_readvariableop_1_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
э
J
.__inference_activation_20_layer_call_fn_435633

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_4323782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_20_layer_call_fn_435718

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4315612
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Е
e
I__inference_activation_23_layer_call_and_return_conditional_losses_436216

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_432256

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435272

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436141

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╝

Б
E__inference_conv2d_10_layer_call_and_return_conditional_losses_432425

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Е
e
I__inference_activation_17_layer_call_and_return_conditional_losses_432230

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:           А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Ї
R
&__inference_add_8_layer_call_fn_436087
inputs_0
inputs_1
identity╪
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_8_layer_call_and_return_conditional_losses_4325442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         ╪:         ╪:Z V
0
_output_shapes
:         ╪
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:         ╪
"
_user_specified_name
inputs/1
р
╥
4__inference_separable_conv2d_18_layer_call_fn_431705

inputs"
unknown:А%
	unknown_0:А╪
	unknown_1:	╪
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_4316932
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436123

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_23_layer_call_fn_436172

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4320472
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Я
Ц
(__inference_dense_2_layer_call_fn_436268

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4326062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╤
c
*__inference_dropout_2_layer_call_fn_436248

inputs
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4327722
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╖

ї
C__inference_dense_2_layer_call_and_return_conditional_losses_432606

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
а├
╦
C__inference_model_2_layer_call_and_return_conditional_losses_433969
input_3*
conv2d_8_433806:А
conv2d_8_433808:	А,
batch_normalization_16_433811:	А,
batch_normalization_16_433813:	А,
batch_normalization_16_433815:	А,
batch_normalization_16_433817:	А5
separable_conv2d_14_433822:А6
separable_conv2d_14_433824:АА)
separable_conv2d_14_433826:	А,
batch_normalization_17_433829:	А,
batch_normalization_17_433831:	А,
batch_normalization_17_433833:	А,
batch_normalization_17_433835:	А5
separable_conv2d_15_433839:А6
separable_conv2d_15_433841:АА)
separable_conv2d_15_433843:	А,
batch_normalization_18_433846:	А,
batch_normalization_18_433848:	А,
batch_normalization_18_433850:	А,
batch_normalization_18_433852:	А+
conv2d_9_433856:АА
conv2d_9_433858:	А5
separable_conv2d_16_433863:А6
separable_conv2d_16_433865:АА)
separable_conv2d_16_433867:	А,
batch_normalization_19_433870:	А,
batch_normalization_19_433872:	А,
batch_normalization_19_433874:	А,
batch_normalization_19_433876:	А5
separable_conv2d_17_433880:А6
separable_conv2d_17_433882:АА)
separable_conv2d_17_433884:	А,
batch_normalization_20_433887:	А,
batch_normalization_20_433889:	А,
batch_normalization_20_433891:	А,
batch_normalization_20_433893:	А,
conv2d_10_433897:АА
conv2d_10_433899:	А5
separable_conv2d_18_433904:А6
separable_conv2d_18_433906:А╪)
separable_conv2d_18_433908:	╪,
batch_normalization_21_433911:	╪,
batch_normalization_21_433913:	╪,
batch_normalization_21_433915:	╪,
batch_normalization_21_433917:	╪5
separable_conv2d_19_433921:╪6
separable_conv2d_19_433923:╪╪)
separable_conv2d_19_433925:	╪,
batch_normalization_22_433928:	╪,
batch_normalization_22_433930:	╪,
batch_normalization_22_433932:	╪,
batch_normalization_22_433934:	╪,
conv2d_11_433938:А╪
conv2d_11_433940:	╪5
separable_conv2d_20_433944:╪6
separable_conv2d_20_433946:╪А)
separable_conv2d_20_433948:	А,
batch_normalization_23_433951:	А,
batch_normalization_23_433953:	А,
batch_normalization_23_433955:	А,
batch_normalization_23_433957:	А!
dense_2_433963:	А

dense_2_433965:

identityИв.batch_normalization_16/StatefulPartitionedCallв.batch_normalization_17/StatefulPartitionedCallв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв+separable_conv2d_16/StatefulPartitionedCallв+separable_conv2d_17/StatefulPartitionedCallв+separable_conv2d_18/StatefulPartitionedCallв+separable_conv2d_19/StatefulPartitionedCallв+separable_conv2d_20/StatefulPartitionedCallm
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_2/Cast/xq
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_2/Cast_1/xЙ
rescaling_2/mulMulinput_3rescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/mulЩ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/addн
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallrescaling_2/add:z:0conv2d_8_433806conv2d_8_433808*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_4321852"
 conv2d_8/StatefulPartitionedCall╦
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_16_433811batch_normalization_16_433813batch_normalization_16_433815batch_normalization_16_433817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_43220820
.batch_normalization_16/StatefulPartitionedCallа
activation_16/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_4322232
activation_16/PartitionedCallП
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_4322302
activation_17/PartitionedCallХ
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0separable_conv2d_14_433822separable_conv2d_14_433824separable_conv2d_14_433826*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_4310532-
+separable_conv2d_14/StatefulPartitionedCall╓
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0batch_normalization_17_433829batch_normalization_17_433831batch_normalization_17_433833batch_normalization_17_433835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_43225620
.batch_normalization_17/StatefulPartitionedCallа
activation_18/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_4322712
activation_18/PartitionedCallХ
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0separable_conv2d_15_433839separable_conv2d_15_433841separable_conv2d_15_433843*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_4312072-
+separable_conv2d_15/StatefulPartitionedCall╓
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_433846batch_normalization_18_433848batch_normalization_18_433850batch_normalization_18_433852*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_43229720
.batch_normalization_18/StatefulPartitionedCallж
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_4313512!
max_pooling2d_6/PartitionedCall└
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_9_433856conv2d_9_433858*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_4323182"
 conv2d_9/StatefulPartitionedCallе
add_6/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_4323302
add_6/PartitionedCallЗ
activation_19/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_4323372
activation_19/PartitionedCallХ
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0separable_conv2d_16_433863separable_conv2d_16_433865separable_conv2d_16_433867*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_4313732-
+separable_conv2d_16/StatefulPartitionedCall╓
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_433870batch_normalization_19_433872batch_normalization_19_433874batch_normalization_19_433876*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_43236320
.batch_normalization_19/StatefulPartitionedCallа
activation_20/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_4323782
activation_20/PartitionedCallХ
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0separable_conv2d_17_433880separable_conv2d_17_433882separable_conv2d_17_433884*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_4315272-
+separable_conv2d_17/StatefulPartitionedCall╓
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_433887batch_normalization_20_433889batch_normalization_20_433891batch_normalization_20_433893*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_43240420
.batch_normalization_20/StatefulPartitionedCallж
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_4316712!
max_pooling2d_7/PartitionedCall╜
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_10_433897conv2d_10_433899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_4324252#
!conv2d_10/StatefulPartitionedCallж
add_7/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_4324372
add_7/PartitionedCallЗ
activation_21/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_4324442
activation_21/PartitionedCallХ
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0separable_conv2d_18_433904separable_conv2d_18_433906separable_conv2d_18_433908*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_4316932-
+separable_conv2d_18/StatefulPartitionedCall╓
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_433911batch_normalization_21_433913batch_normalization_21_433915batch_normalization_21_433917*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_43247020
.batch_normalization_21/StatefulPartitionedCallа
activation_22/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_4324852
activation_22/PartitionedCallХ
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0separable_conv2d_19_433921separable_conv2d_19_433923separable_conv2d_19_433925*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_4318472-
+separable_conv2d_19/StatefulPartitionedCall╓
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_433928batch_normalization_22_433930batch_normalization_22_433932batch_normalization_22_433934*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_43251120
.batch_normalization_22/StatefulPartitionedCallж
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_4319912!
max_pooling2d_8/PartitionedCall╜
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCalladd_7/PartitionedCall:output:0conv2d_11_433938conv2d_11_433940*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_4325322#
!conv2d_11/StatefulPartitionedCallж
add_8/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_8_layer_call_and_return_conditional_losses_4325442
add_8/PartitionedCallН
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCalladd_8/PartitionedCall:output:0separable_conv2d_20_433944separable_conv2d_20_433946separable_conv2d_20_433948*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_4320132-
+separable_conv2d_20/StatefulPartitionedCall╓
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_433951batch_normalization_23_433953batch_normalization_23_433955batch_normalization_23_433957*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_43257020
.batch_normalization_23/StatefulPartitionedCallа
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_4325852
activation_23/PartitionedCallо
*global_average_pooling2d_2/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_4321582,
*global_average_pooling2d_2/PartitionedCallИ
dropout_2/PartitionedCallPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4325932
dropout_2/PartitionedCallо
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_433963dense_2_433965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4326062!
dense_2/StatefulPartitionedCallЎ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall:X T
/
_output_shapes
:         @@
!
_user_specified_name	input_3
╨
┼
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435535

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_432208

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Е
e
I__inference_activation_23_layer_call_and_return_conditional_losses_432585

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
└─
я
C__inference_model_2_layer_call_and_return_conditional_losses_434139
input_3*
conv2d_8_433976:А
conv2d_8_433978:	А,
batch_normalization_16_433981:	А,
batch_normalization_16_433983:	А,
batch_normalization_16_433985:	А,
batch_normalization_16_433987:	А5
separable_conv2d_14_433992:А6
separable_conv2d_14_433994:АА)
separable_conv2d_14_433996:	А,
batch_normalization_17_433999:	А,
batch_normalization_17_434001:	А,
batch_normalization_17_434003:	А,
batch_normalization_17_434005:	А5
separable_conv2d_15_434009:А6
separable_conv2d_15_434011:АА)
separable_conv2d_15_434013:	А,
batch_normalization_18_434016:	А,
batch_normalization_18_434018:	А,
batch_normalization_18_434020:	А,
batch_normalization_18_434022:	А+
conv2d_9_434026:АА
conv2d_9_434028:	А5
separable_conv2d_16_434033:А6
separable_conv2d_16_434035:АА)
separable_conv2d_16_434037:	А,
batch_normalization_19_434040:	А,
batch_normalization_19_434042:	А,
batch_normalization_19_434044:	А,
batch_normalization_19_434046:	А5
separable_conv2d_17_434050:А6
separable_conv2d_17_434052:АА)
separable_conv2d_17_434054:	А,
batch_normalization_20_434057:	А,
batch_normalization_20_434059:	А,
batch_normalization_20_434061:	А,
batch_normalization_20_434063:	А,
conv2d_10_434067:АА
conv2d_10_434069:	А5
separable_conv2d_18_434074:А6
separable_conv2d_18_434076:А╪)
separable_conv2d_18_434078:	╪,
batch_normalization_21_434081:	╪,
batch_normalization_21_434083:	╪,
batch_normalization_21_434085:	╪,
batch_normalization_21_434087:	╪5
separable_conv2d_19_434091:╪6
separable_conv2d_19_434093:╪╪)
separable_conv2d_19_434095:	╪,
batch_normalization_22_434098:	╪,
batch_normalization_22_434100:	╪,
batch_normalization_22_434102:	╪,
batch_normalization_22_434104:	╪,
conv2d_11_434108:А╪
conv2d_11_434110:	╪5
separable_conv2d_20_434114:╪6
separable_conv2d_20_434116:╪А)
separable_conv2d_20_434118:	А,
batch_normalization_23_434121:	А,
batch_normalization_23_434123:	А,
batch_normalization_23_434125:	А,
batch_normalization_23_434127:	А!
dense_2_434133:	А

dense_2_434135:

identityИв.batch_normalization_16/StatefulPartitionedCallв.batch_normalization_17/StatefulPartitionedCallв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв+separable_conv2d_16/StatefulPartitionedCallв+separable_conv2d_17/StatefulPartitionedCallв+separable_conv2d_18/StatefulPartitionedCallв+separable_conv2d_19/StatefulPartitionedCallв+separable_conv2d_20/StatefulPartitionedCallm
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_2/Cast/xq
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_2/Cast_1/xЙ
rescaling_2/mulMulinput_3rescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/mulЩ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/addн
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallrescaling_2/add:z:0conv2d_8_433976conv2d_8_433978*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_4321852"
 conv2d_8/StatefulPartitionedCall╔
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_16_433981batch_normalization_16_433983batch_normalization_16_433985batch_normalization_16_433987*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_43321520
.batch_normalization_16/StatefulPartitionedCallа
activation_16/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_4322232
activation_16/PartitionedCallП
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_4322302
activation_17/PartitionedCallХ
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0separable_conv2d_14_433992separable_conv2d_14_433994separable_conv2d_14_433996*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_4310532-
+separable_conv2d_14/StatefulPartitionedCall╘
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0batch_normalization_17_433999batch_normalization_17_434001batch_normalization_17_434003batch_normalization_17_434005*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_43315920
.batch_normalization_17/StatefulPartitionedCallа
activation_18/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_4322712
activation_18/PartitionedCallХ
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0separable_conv2d_15_434009separable_conv2d_15_434011separable_conv2d_15_434013*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_4312072-
+separable_conv2d_15/StatefulPartitionedCall╘
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_434016batch_normalization_18_434018batch_normalization_18_434020batch_normalization_18_434022*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_43310920
.batch_normalization_18/StatefulPartitionedCallж
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_4313512!
max_pooling2d_6/PartitionedCall└
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_9_434026conv2d_9_434028*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_4323182"
 conv2d_9/StatefulPartitionedCallе
add_6/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_4323302
add_6/PartitionedCallЗ
activation_19/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_4323372
activation_19/PartitionedCallХ
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0separable_conv2d_16_434033separable_conv2d_16_434035separable_conv2d_16_434037*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_4313732-
+separable_conv2d_16/StatefulPartitionedCall╘
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_434040batch_normalization_19_434042batch_normalization_19_434044batch_normalization_19_434046*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_43304220
.batch_normalization_19/StatefulPartitionedCallа
activation_20/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_4323782
activation_20/PartitionedCallХ
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0separable_conv2d_17_434050separable_conv2d_17_434052separable_conv2d_17_434054*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_4315272-
+separable_conv2d_17/StatefulPartitionedCall╘
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_434057batch_normalization_20_434059batch_normalization_20_434061batch_normalization_20_434063*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_43299220
.batch_normalization_20/StatefulPartitionedCallж
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_4316712!
max_pooling2d_7/PartitionedCall╜
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_10_434067conv2d_10_434069*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_4324252#
!conv2d_10/StatefulPartitionedCallж
add_7/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_4324372
add_7/PartitionedCallЗ
activation_21/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_4324442
activation_21/PartitionedCallХ
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0separable_conv2d_18_434074separable_conv2d_18_434076separable_conv2d_18_434078*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_4316932-
+separable_conv2d_18/StatefulPartitionedCall╘
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_434081batch_normalization_21_434083batch_normalization_21_434085batch_normalization_21_434087*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_43292520
.batch_normalization_21/StatefulPartitionedCallа
activation_22/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_4324852
activation_22/PartitionedCallХ
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0separable_conv2d_19_434091separable_conv2d_19_434093separable_conv2d_19_434095*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_4318472-
+separable_conv2d_19/StatefulPartitionedCall╘
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_434098batch_normalization_22_434100batch_normalization_22_434102batch_normalization_22_434104*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_43287520
.batch_normalization_22/StatefulPartitionedCallж
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_4319912!
max_pooling2d_8/PartitionedCall╜
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCalladd_7/PartitionedCall:output:0conv2d_11_434108conv2d_11_434110*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_4325322#
!conv2d_11/StatefulPartitionedCallж
add_8/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_8_layer_call_and_return_conditional_losses_4325442
add_8/PartitionedCallН
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCalladd_8/PartitionedCall:output:0separable_conv2d_20_434114separable_conv2d_20_434116separable_conv2d_20_434118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_4320132-
+separable_conv2d_20/StatefulPartitionedCall╘
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_434121batch_normalization_23_434123batch_normalization_23_434125batch_normalization_23_434127*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_43281420
.batch_normalization_23/StatefulPartitionedCallа
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_4325852
activation_23/PartitionedCallо
*global_average_pooling2d_2/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_4321582,
*global_average_pooling2d_2/PartitionedCallа
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4327722#
!dropout_2/StatefulPartitionedCall╢
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_434133dense_2_434135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4326062!
dense_2/StatefulPartitionedCallЪ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall:X T
/
_output_shapes
:         @@
!
_user_specified_name	input_3
З
m
A__inference_add_7_layer_call_and_return_conditional_losses_435782
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:         А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:Z V
0
_output_shapes
:         А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:         А
"
_user_specified_name
inputs/1
Ь
б
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435816

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
┤
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_436238

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Е
e
I__inference_activation_21_layer_call_and_return_conditional_losses_435793

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
р
╥
4__inference_separable_conv2d_19_layer_call_fn_431859

inputs"
unknown:╪%
	unknown_0:╪╪
	unknown_1:	╪
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_4318472
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ╪: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
▐ы
╤z
"__inference__traced_restore_437297
file_prefix;
 assignvariableop_conv2d_8_kernel:А/
 assignvariableop_1_conv2d_8_bias:	А>
/assignvariableop_2_batch_normalization_16_gamma:	А=
.assignvariableop_3_batch_normalization_16_beta:	АD
5assignvariableop_4_batch_normalization_16_moving_mean:	АH
9assignvariableop_5_batch_normalization_16_moving_variance:	АR
7assignvariableop_6_separable_conv2d_14_depthwise_kernel:АS
7assignvariableop_7_separable_conv2d_14_pointwise_kernel:АА:
+assignvariableop_8_separable_conv2d_14_bias:	А>
/assignvariableop_9_batch_normalization_17_gamma:	А>
/assignvariableop_10_batch_normalization_17_beta:	АE
6assignvariableop_11_batch_normalization_17_moving_mean:	АI
:assignvariableop_12_batch_normalization_17_moving_variance:	АS
8assignvariableop_13_separable_conv2d_15_depthwise_kernel:АT
8assignvariableop_14_separable_conv2d_15_pointwise_kernel:АА;
,assignvariableop_15_separable_conv2d_15_bias:	А?
0assignvariableop_16_batch_normalization_18_gamma:	А>
/assignvariableop_17_batch_normalization_18_beta:	АE
6assignvariableop_18_batch_normalization_18_moving_mean:	АI
:assignvariableop_19_batch_normalization_18_moving_variance:	А?
#assignvariableop_20_conv2d_9_kernel:АА0
!assignvariableop_21_conv2d_9_bias:	АS
8assignvariableop_22_separable_conv2d_16_depthwise_kernel:АT
8assignvariableop_23_separable_conv2d_16_pointwise_kernel:АА;
,assignvariableop_24_separable_conv2d_16_bias:	А?
0assignvariableop_25_batch_normalization_19_gamma:	А>
/assignvariableop_26_batch_normalization_19_beta:	АE
6assignvariableop_27_batch_normalization_19_moving_mean:	АI
:assignvariableop_28_batch_normalization_19_moving_variance:	АS
8assignvariableop_29_separable_conv2d_17_depthwise_kernel:АT
8assignvariableop_30_separable_conv2d_17_pointwise_kernel:АА;
,assignvariableop_31_separable_conv2d_17_bias:	А?
0assignvariableop_32_batch_normalization_20_gamma:	А>
/assignvariableop_33_batch_normalization_20_beta:	АE
6assignvariableop_34_batch_normalization_20_moving_mean:	АI
:assignvariableop_35_batch_normalization_20_moving_variance:	А@
$assignvariableop_36_conv2d_10_kernel:АА1
"assignvariableop_37_conv2d_10_bias:	АS
8assignvariableop_38_separable_conv2d_18_depthwise_kernel:АT
8assignvariableop_39_separable_conv2d_18_pointwise_kernel:А╪;
,assignvariableop_40_separable_conv2d_18_bias:	╪?
0assignvariableop_41_batch_normalization_21_gamma:	╪>
/assignvariableop_42_batch_normalization_21_beta:	╪E
6assignvariableop_43_batch_normalization_21_moving_mean:	╪I
:assignvariableop_44_batch_normalization_21_moving_variance:	╪S
8assignvariableop_45_separable_conv2d_19_depthwise_kernel:╪T
8assignvariableop_46_separable_conv2d_19_pointwise_kernel:╪╪;
,assignvariableop_47_separable_conv2d_19_bias:	╪?
0assignvariableop_48_batch_normalization_22_gamma:	╪>
/assignvariableop_49_batch_normalization_22_beta:	╪E
6assignvariableop_50_batch_normalization_22_moving_mean:	╪I
:assignvariableop_51_batch_normalization_22_moving_variance:	╪@
$assignvariableop_52_conv2d_11_kernel:А╪1
"assignvariableop_53_conv2d_11_bias:	╪S
8assignvariableop_54_separable_conv2d_20_depthwise_kernel:╪T
8assignvariableop_55_separable_conv2d_20_pointwise_kernel:╪А;
,assignvariableop_56_separable_conv2d_20_bias:	А?
0assignvariableop_57_batch_normalization_23_gamma:	А>
/assignvariableop_58_batch_normalization_23_beta:	АE
6assignvariableop_59_batch_normalization_23_moving_mean:	АI
:assignvariableop_60_batch_normalization_23_moving_variance:	А5
"assignvariableop_61_dense_2_kernel:	А
.
 assignvariableop_62_dense_2_bias:
'
assignvariableop_63_adam_iter:	 )
assignvariableop_64_adam_beta_1: )
assignvariableop_65_adam_beta_2: (
assignvariableop_66_adam_decay: 0
&assignvariableop_67_adam_learning_rate: #
assignvariableop_68_total: #
assignvariableop_69_count: %
assignvariableop_70_total_1: %
assignvariableop_71_count_1: E
*assignvariableop_72_adam_conv2d_8_kernel_m:А7
(assignvariableop_73_adam_conv2d_8_bias_m:	АF
7assignvariableop_74_adam_batch_normalization_16_gamma_m:	АE
6assignvariableop_75_adam_batch_normalization_16_beta_m:	АZ
?assignvariableop_76_adam_separable_conv2d_14_depthwise_kernel_m:А[
?assignvariableop_77_adam_separable_conv2d_14_pointwise_kernel_m:ААB
3assignvariableop_78_adam_separable_conv2d_14_bias_m:	АF
7assignvariableop_79_adam_batch_normalization_17_gamma_m:	АE
6assignvariableop_80_adam_batch_normalization_17_beta_m:	АZ
?assignvariableop_81_adam_separable_conv2d_15_depthwise_kernel_m:А[
?assignvariableop_82_adam_separable_conv2d_15_pointwise_kernel_m:ААB
3assignvariableop_83_adam_separable_conv2d_15_bias_m:	АF
7assignvariableop_84_adam_batch_normalization_18_gamma_m:	АE
6assignvariableop_85_adam_batch_normalization_18_beta_m:	АF
*assignvariableop_86_adam_conv2d_9_kernel_m:АА7
(assignvariableop_87_adam_conv2d_9_bias_m:	АZ
?assignvariableop_88_adam_separable_conv2d_16_depthwise_kernel_m:А[
?assignvariableop_89_adam_separable_conv2d_16_pointwise_kernel_m:ААB
3assignvariableop_90_adam_separable_conv2d_16_bias_m:	АF
7assignvariableop_91_adam_batch_normalization_19_gamma_m:	АE
6assignvariableop_92_adam_batch_normalization_19_beta_m:	АZ
?assignvariableop_93_adam_separable_conv2d_17_depthwise_kernel_m:А[
?assignvariableop_94_adam_separable_conv2d_17_pointwise_kernel_m:ААB
3assignvariableop_95_adam_separable_conv2d_17_bias_m:	АF
7assignvariableop_96_adam_batch_normalization_20_gamma_m:	АE
6assignvariableop_97_adam_batch_normalization_20_beta_m:	АG
+assignvariableop_98_adam_conv2d_10_kernel_m:АА8
)assignvariableop_99_adam_conv2d_10_bias_m:	А[
@assignvariableop_100_adam_separable_conv2d_18_depthwise_kernel_m:А\
@assignvariableop_101_adam_separable_conv2d_18_pointwise_kernel_m:А╪C
4assignvariableop_102_adam_separable_conv2d_18_bias_m:	╪G
8assignvariableop_103_adam_batch_normalization_21_gamma_m:	╪F
7assignvariableop_104_adam_batch_normalization_21_beta_m:	╪[
@assignvariableop_105_adam_separable_conv2d_19_depthwise_kernel_m:╪\
@assignvariableop_106_adam_separable_conv2d_19_pointwise_kernel_m:╪╪C
4assignvariableop_107_adam_separable_conv2d_19_bias_m:	╪G
8assignvariableop_108_adam_batch_normalization_22_gamma_m:	╪F
7assignvariableop_109_adam_batch_normalization_22_beta_m:	╪H
,assignvariableop_110_adam_conv2d_11_kernel_m:А╪9
*assignvariableop_111_adam_conv2d_11_bias_m:	╪[
@assignvariableop_112_adam_separable_conv2d_20_depthwise_kernel_m:╪\
@assignvariableop_113_adam_separable_conv2d_20_pointwise_kernel_m:╪АC
4assignvariableop_114_adam_separable_conv2d_20_bias_m:	АG
8assignvariableop_115_adam_batch_normalization_23_gamma_m:	АF
7assignvariableop_116_adam_batch_normalization_23_beta_m:	А=
*assignvariableop_117_adam_dense_2_kernel_m:	А
6
(assignvariableop_118_adam_dense_2_bias_m:
F
+assignvariableop_119_adam_conv2d_8_kernel_v:А8
)assignvariableop_120_adam_conv2d_8_bias_v:	АG
8assignvariableop_121_adam_batch_normalization_16_gamma_v:	АF
7assignvariableop_122_adam_batch_normalization_16_beta_v:	А[
@assignvariableop_123_adam_separable_conv2d_14_depthwise_kernel_v:А\
@assignvariableop_124_adam_separable_conv2d_14_pointwise_kernel_v:ААC
4assignvariableop_125_adam_separable_conv2d_14_bias_v:	АG
8assignvariableop_126_adam_batch_normalization_17_gamma_v:	АF
7assignvariableop_127_adam_batch_normalization_17_beta_v:	А[
@assignvariableop_128_adam_separable_conv2d_15_depthwise_kernel_v:А\
@assignvariableop_129_adam_separable_conv2d_15_pointwise_kernel_v:ААC
4assignvariableop_130_adam_separable_conv2d_15_bias_v:	АG
8assignvariableop_131_adam_batch_normalization_18_gamma_v:	АF
7assignvariableop_132_adam_batch_normalization_18_beta_v:	АG
+assignvariableop_133_adam_conv2d_9_kernel_v:АА8
)assignvariableop_134_adam_conv2d_9_bias_v:	А[
@assignvariableop_135_adam_separable_conv2d_16_depthwise_kernel_v:А\
@assignvariableop_136_adam_separable_conv2d_16_pointwise_kernel_v:ААC
4assignvariableop_137_adam_separable_conv2d_16_bias_v:	АG
8assignvariableop_138_adam_batch_normalization_19_gamma_v:	АF
7assignvariableop_139_adam_batch_normalization_19_beta_v:	А[
@assignvariableop_140_adam_separable_conv2d_17_depthwise_kernel_v:А\
@assignvariableop_141_adam_separable_conv2d_17_pointwise_kernel_v:ААC
4assignvariableop_142_adam_separable_conv2d_17_bias_v:	АG
8assignvariableop_143_adam_batch_normalization_20_gamma_v:	АF
7assignvariableop_144_adam_batch_normalization_20_beta_v:	АH
,assignvariableop_145_adam_conv2d_10_kernel_v:АА9
*assignvariableop_146_adam_conv2d_10_bias_v:	А[
@assignvariableop_147_adam_separable_conv2d_18_depthwise_kernel_v:А\
@assignvariableop_148_adam_separable_conv2d_18_pointwise_kernel_v:А╪C
4assignvariableop_149_adam_separable_conv2d_18_bias_v:	╪G
8assignvariableop_150_adam_batch_normalization_21_gamma_v:	╪F
7assignvariableop_151_adam_batch_normalization_21_beta_v:	╪[
@assignvariableop_152_adam_separable_conv2d_19_depthwise_kernel_v:╪\
@assignvariableop_153_adam_separable_conv2d_19_pointwise_kernel_v:╪╪C
4assignvariableop_154_adam_separable_conv2d_19_bias_v:	╪G
8assignvariableop_155_adam_batch_normalization_22_gamma_v:	╪F
7assignvariableop_156_adam_batch_normalization_22_beta_v:	╪H
,assignvariableop_157_adam_conv2d_11_kernel_v:А╪9
*assignvariableop_158_adam_conv2d_11_bias_v:	╪[
@assignvariableop_159_adam_separable_conv2d_20_depthwise_kernel_v:╪\
@assignvariableop_160_adam_separable_conv2d_20_pointwise_kernel_v:╪АC
4assignvariableop_161_adam_separable_conv2d_20_bias_v:	АG
8assignvariableop_162_adam_batch_normalization_23_gamma_v:	АF
7assignvariableop_163_adam_batch_normalization_23_beta_v:	А=
*assignvariableop_164_adam_dense_2_kernel_v:	А
6
(assignvariableop_165_adam_dense_2_bias_v:

identity_167ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_110вAssignVariableOp_111вAssignVariableOp_112вAssignVariableOp_113вAssignVariableOp_114вAssignVariableOp_115вAssignVariableOp_116вAssignVariableOp_117вAssignVariableOp_118вAssignVariableOp_119вAssignVariableOp_12вAssignVariableOp_120вAssignVariableOp_121вAssignVariableOp_122вAssignVariableOp_123вAssignVariableOp_124вAssignVariableOp_125вAssignVariableOp_126вAssignVariableOp_127вAssignVariableOp_128вAssignVariableOp_129вAssignVariableOp_13вAssignVariableOp_130вAssignVariableOp_131вAssignVariableOp_132вAssignVariableOp_133вAssignVariableOp_134вAssignVariableOp_135вAssignVariableOp_136вAssignVariableOp_137вAssignVariableOp_138вAssignVariableOp_139вAssignVariableOp_14вAssignVariableOp_140вAssignVariableOp_141вAssignVariableOp_142вAssignVariableOp_143вAssignVariableOp_144вAssignVariableOp_145вAssignVariableOp_146вAssignVariableOp_147вAssignVariableOp_148вAssignVariableOp_149вAssignVariableOp_15вAssignVariableOp_150вAssignVariableOp_151вAssignVariableOp_152вAssignVariableOp_153вAssignVariableOp_154вAssignVariableOp_155вAssignVariableOp_156вAssignVariableOp_157вAssignVariableOp_158вAssignVariableOp_159вAssignVariableOp_16вAssignVariableOp_160вAssignVariableOp_161вAssignVariableOp_162вAssignVariableOp_163вAssignVariableOp_164вAssignVariableOp_165вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99╫a
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:з*
dtype0*т`
value╪`B╒`зB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesс
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:з*
dtype0*ф
value┌B╫зB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesї
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▓
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*╕
dtypesн
к2з	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2┤
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_16_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3│
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_16_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4║
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_16_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╛
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_16_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╝
AssignVariableOp_6AssignVariableOp7assignvariableop_6_separable_conv2d_14_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╝
AssignVariableOp_7AssignVariableOp7assignvariableop_7_separable_conv2d_14_pointwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8░
AssignVariableOp_8AssignVariableOp+assignvariableop_8_separable_conv2d_14_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9┤
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_17_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╖
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_17_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╛
AssignVariableOp_11AssignVariableOp6assignvariableop_11_batch_normalization_17_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12┬
AssignVariableOp_12AssignVariableOp:assignvariableop_12_batch_normalization_17_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13└
AssignVariableOp_13AssignVariableOp8assignvariableop_13_separable_conv2d_15_depthwise_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14└
AssignVariableOp_14AssignVariableOp8assignvariableop_14_separable_conv2d_15_pointwise_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┤
AssignVariableOp_15AssignVariableOp,assignvariableop_15_separable_conv2d_15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╕
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_18_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╖
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_18_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╛
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_18_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┬
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_18_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20л
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_9_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21й
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_9_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22└
AssignVariableOp_22AssignVariableOp8assignvariableop_22_separable_conv2d_16_depthwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23└
AssignVariableOp_23AssignVariableOp8assignvariableop_23_separable_conv2d_16_pointwise_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┤
AssignVariableOp_24AssignVariableOp,assignvariableop_24_separable_conv2d_16_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╕
AssignVariableOp_25AssignVariableOp0assignvariableop_25_batch_normalization_19_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╖
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_19_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_batch_normalization_19_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28┬
AssignVariableOp_28AssignVariableOp:assignvariableop_28_batch_normalization_19_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29└
AssignVariableOp_29AssignVariableOp8assignvariableop_29_separable_conv2d_17_depthwise_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30└
AssignVariableOp_30AssignVariableOp8assignvariableop_30_separable_conv2d_17_pointwise_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31┤
AssignVariableOp_31AssignVariableOp,assignvariableop_31_separable_conv2d_17_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╕
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_20_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╖
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_20_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╛
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_20_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┬
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_20_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36м
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_10_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37к
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_10_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38└
AssignVariableOp_38AssignVariableOp8assignvariableop_38_separable_conv2d_18_depthwise_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39└
AssignVariableOp_39AssignVariableOp8assignvariableop_39_separable_conv2d_18_pointwise_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40┤
AssignVariableOp_40AssignVariableOp,assignvariableop_40_separable_conv2d_18_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╕
AssignVariableOp_41AssignVariableOp0assignvariableop_41_batch_normalization_21_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╖
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_21_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╛
AssignVariableOp_43AssignVariableOp6assignvariableop_43_batch_normalization_21_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44┬
AssignVariableOp_44AssignVariableOp:assignvariableop_44_batch_normalization_21_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45└
AssignVariableOp_45AssignVariableOp8assignvariableop_45_separable_conv2d_19_depthwise_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46└
AssignVariableOp_46AssignVariableOp8assignvariableop_46_separable_conv2d_19_pointwise_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47┤
AssignVariableOp_47AssignVariableOp,assignvariableop_47_separable_conv2d_19_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48╕
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_22_gammaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49╖
AssignVariableOp_49AssignVariableOp/assignvariableop_49_batch_normalization_22_betaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50╛
AssignVariableOp_50AssignVariableOp6assignvariableop_50_batch_normalization_22_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51┬
AssignVariableOp_51AssignVariableOp:assignvariableop_51_batch_normalization_22_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52м
AssignVariableOp_52AssignVariableOp$assignvariableop_52_conv2d_11_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53к
AssignVariableOp_53AssignVariableOp"assignvariableop_53_conv2d_11_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54└
AssignVariableOp_54AssignVariableOp8assignvariableop_54_separable_conv2d_20_depthwise_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55└
AssignVariableOp_55AssignVariableOp8assignvariableop_55_separable_conv2d_20_pointwise_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56┤
AssignVariableOp_56AssignVariableOp,assignvariableop_56_separable_conv2d_20_biasIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57╕
AssignVariableOp_57AssignVariableOp0assignvariableop_57_batch_normalization_23_gammaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58╖
AssignVariableOp_58AssignVariableOp/assignvariableop_58_batch_normalization_23_betaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59╛
AssignVariableOp_59AssignVariableOp6assignvariableop_59_batch_normalization_23_moving_meanIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60┬
AssignVariableOp_60AssignVariableOp:assignvariableop_60_batch_normalization_23_moving_varianceIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61к
AssignVariableOp_61AssignVariableOp"assignvariableop_61_dense_2_kernelIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62и
AssignVariableOp_62AssignVariableOp assignvariableop_62_dense_2_biasIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_63е
AssignVariableOp_63AssignVariableOpassignvariableop_63_adam_iterIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64з
AssignVariableOp_64AssignVariableOpassignvariableop_64_adam_beta_1Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65з
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_beta_2Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66ж
AssignVariableOp_66AssignVariableOpassignvariableop_66_adam_decayIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67о
AssignVariableOp_67AssignVariableOp&assignvariableop_67_adam_learning_rateIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68б
AssignVariableOp_68AssignVariableOpassignvariableop_68_totalIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69б
AssignVariableOp_69AssignVariableOpassignvariableop_69_countIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70г
AssignVariableOp_70AssignVariableOpassignvariableop_70_total_1Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71г
AssignVariableOp_71AssignVariableOpassignvariableop_71_count_1Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72▓
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_8_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73░
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_conv2d_8_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74┐
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_16_gamma_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75╛
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_batch_normalization_16_beta_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76╟
AssignVariableOp_76AssignVariableOp?assignvariableop_76_adam_separable_conv2d_14_depthwise_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77╟
AssignVariableOp_77AssignVariableOp?assignvariableop_77_adam_separable_conv2d_14_pointwise_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78╗
AssignVariableOp_78AssignVariableOp3assignvariableop_78_adam_separable_conv2d_14_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79┐
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_17_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80╛
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_17_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81╟
AssignVariableOp_81AssignVariableOp?assignvariableop_81_adam_separable_conv2d_15_depthwise_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82╟
AssignVariableOp_82AssignVariableOp?assignvariableop_82_adam_separable_conv2d_15_pointwise_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83╗
AssignVariableOp_83AssignVariableOp3assignvariableop_83_adam_separable_conv2d_15_bias_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84┐
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_18_gamma_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85╛
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_18_beta_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86▓
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_conv2d_9_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87░
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_conv2d_9_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88╟
AssignVariableOp_88AssignVariableOp?assignvariableop_88_adam_separable_conv2d_16_depthwise_kernel_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89╟
AssignVariableOp_89AssignVariableOp?assignvariableop_89_adam_separable_conv2d_16_pointwise_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90╗
AssignVariableOp_90AssignVariableOp3assignvariableop_90_adam_separable_conv2d_16_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91┐
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_19_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92╛
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_19_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93╟
AssignVariableOp_93AssignVariableOp?assignvariableop_93_adam_separable_conv2d_17_depthwise_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94╟
AssignVariableOp_94AssignVariableOp?assignvariableop_94_adam_separable_conv2d_17_pointwise_kernel_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95╗
AssignVariableOp_95AssignVariableOp3assignvariableop_95_adam_separable_conv2d_17_bias_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96┐
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_20_gamma_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97╛
AssignVariableOp_97AssignVariableOp6assignvariableop_97_adam_batch_normalization_20_beta_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98│
AssignVariableOp_98AssignVariableOp+assignvariableop_98_adam_conv2d_10_kernel_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99▒
AssignVariableOp_99AssignVariableOp)assignvariableop_99_adam_conv2d_10_bias_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100╦
AssignVariableOp_100AssignVariableOp@assignvariableop_100_adam_separable_conv2d_18_depthwise_kernel_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101╦
AssignVariableOp_101AssignVariableOp@assignvariableop_101_adam_separable_conv2d_18_pointwise_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102┐
AssignVariableOp_102AssignVariableOp4assignvariableop_102_adam_separable_conv2d_18_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103├
AssignVariableOp_103AssignVariableOp8assignvariableop_103_adam_batch_normalization_21_gamma_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104┬
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_21_beta_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105╦
AssignVariableOp_105AssignVariableOp@assignvariableop_105_adam_separable_conv2d_19_depthwise_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106╦
AssignVariableOp_106AssignVariableOp@assignvariableop_106_adam_separable_conv2d_19_pointwise_kernel_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107┐
AssignVariableOp_107AssignVariableOp4assignvariableop_107_adam_separable_conv2d_19_bias_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108├
AssignVariableOp_108AssignVariableOp8assignvariableop_108_adam_batch_normalization_22_gamma_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109┬
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_batch_normalization_22_beta_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110╖
AssignVariableOp_110AssignVariableOp,assignvariableop_110_adam_conv2d_11_kernel_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111╡
AssignVariableOp_111AssignVariableOp*assignvariableop_111_adam_conv2d_11_bias_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112╦
AssignVariableOp_112AssignVariableOp@assignvariableop_112_adam_separable_conv2d_20_depthwise_kernel_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113╦
AssignVariableOp_113AssignVariableOp@assignvariableop_113_adam_separable_conv2d_20_pointwise_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114┐
AssignVariableOp_114AssignVariableOp4assignvariableop_114_adam_separable_conv2d_20_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115├
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_23_gamma_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116┬
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_23_beta_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117╡
AssignVariableOp_117AssignVariableOp*assignvariableop_117_adam_dense_2_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118│
AssignVariableOp_118AssignVariableOp(assignvariableop_118_adam_dense_2_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119╢
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_conv2d_8_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120┤
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_conv2d_8_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121├
AssignVariableOp_121AssignVariableOp8assignvariableop_121_adam_batch_normalization_16_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122┬
AssignVariableOp_122AssignVariableOp7assignvariableop_122_adam_batch_normalization_16_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123╦
AssignVariableOp_123AssignVariableOp@assignvariableop_123_adam_separable_conv2d_14_depthwise_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124╦
AssignVariableOp_124AssignVariableOp@assignvariableop_124_adam_separable_conv2d_14_pointwise_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125┐
AssignVariableOp_125AssignVariableOp4assignvariableop_125_adam_separable_conv2d_14_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126├
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_17_gamma_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127┬
AssignVariableOp_127AssignVariableOp7assignvariableop_127_adam_batch_normalization_17_beta_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128╦
AssignVariableOp_128AssignVariableOp@assignvariableop_128_adam_separable_conv2d_15_depthwise_kernel_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129╦
AssignVariableOp_129AssignVariableOp@assignvariableop_129_adam_separable_conv2d_15_pointwise_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130┐
AssignVariableOp_130AssignVariableOp4assignvariableop_130_adam_separable_conv2d_15_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131├
AssignVariableOp_131AssignVariableOp8assignvariableop_131_adam_batch_normalization_18_gamma_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132┬
AssignVariableOp_132AssignVariableOp7assignvariableop_132_adam_batch_normalization_18_beta_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133╢
AssignVariableOp_133AssignVariableOp+assignvariableop_133_adam_conv2d_9_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134┤
AssignVariableOp_134AssignVariableOp)assignvariableop_134_adam_conv2d_9_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135╦
AssignVariableOp_135AssignVariableOp@assignvariableop_135_adam_separable_conv2d_16_depthwise_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136╦
AssignVariableOp_136AssignVariableOp@assignvariableop_136_adam_separable_conv2d_16_pointwise_kernel_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137┐
AssignVariableOp_137AssignVariableOp4assignvariableop_137_adam_separable_conv2d_16_bias_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138├
AssignVariableOp_138AssignVariableOp8assignvariableop_138_adam_batch_normalization_19_gamma_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139┬
AssignVariableOp_139AssignVariableOp7assignvariableop_139_adam_batch_normalization_19_beta_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140╦
AssignVariableOp_140AssignVariableOp@assignvariableop_140_adam_separable_conv2d_17_depthwise_kernel_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141╦
AssignVariableOp_141AssignVariableOp@assignvariableop_141_adam_separable_conv2d_17_pointwise_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142┐
AssignVariableOp_142AssignVariableOp4assignvariableop_142_adam_separable_conv2d_17_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143├
AssignVariableOp_143AssignVariableOp8assignvariableop_143_adam_batch_normalization_20_gamma_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144┬
AssignVariableOp_144AssignVariableOp7assignvariableop_144_adam_batch_normalization_20_beta_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145╖
AssignVariableOp_145AssignVariableOp,assignvariableop_145_adam_conv2d_10_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146╡
AssignVariableOp_146AssignVariableOp*assignvariableop_146_adam_conv2d_10_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147╦
AssignVariableOp_147AssignVariableOp@assignvariableop_147_adam_separable_conv2d_18_depthwise_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148╦
AssignVariableOp_148AssignVariableOp@assignvariableop_148_adam_separable_conv2d_18_pointwise_kernel_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149┐
AssignVariableOp_149AssignVariableOp4assignvariableop_149_adam_separable_conv2d_18_bias_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150├
AssignVariableOp_150AssignVariableOp8assignvariableop_150_adam_batch_normalization_21_gamma_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151┬
AssignVariableOp_151AssignVariableOp7assignvariableop_151_adam_batch_normalization_21_beta_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152╦
AssignVariableOp_152AssignVariableOp@assignvariableop_152_adam_separable_conv2d_19_depthwise_kernel_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_152q
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:2
Identity_153╦
AssignVariableOp_153AssignVariableOp@assignvariableop_153_adam_separable_conv2d_19_pointwise_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_153q
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:2
Identity_154┐
AssignVariableOp_154AssignVariableOp4assignvariableop_154_adam_separable_conv2d_19_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_154q
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:2
Identity_155├
AssignVariableOp_155AssignVariableOp8assignvariableop_155_adam_batch_normalization_22_gamma_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_155q
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:2
Identity_156┬
AssignVariableOp_156AssignVariableOp7assignvariableop_156_adam_batch_normalization_22_beta_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_156q
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:2
Identity_157╖
AssignVariableOp_157AssignVariableOp,assignvariableop_157_adam_conv2d_11_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_157q
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:2
Identity_158╡
AssignVariableOp_158AssignVariableOp*assignvariableop_158_adam_conv2d_11_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_158q
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:2
Identity_159╦
AssignVariableOp_159AssignVariableOp@assignvariableop_159_adam_separable_conv2d_20_depthwise_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159q
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:2
Identity_160╦
AssignVariableOp_160AssignVariableOp@assignvariableop_160_adam_separable_conv2d_20_pointwise_kernel_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_160q
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:2
Identity_161┐
AssignVariableOp_161AssignVariableOp4assignvariableop_161_adam_separable_conv2d_20_bias_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_161q
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:2
Identity_162├
AssignVariableOp_162AssignVariableOp8assignvariableop_162_adam_batch_normalization_23_gamma_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_162q
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:2
Identity_163┬
AssignVariableOp_163AssignVariableOp7assignvariableop_163_adam_batch_normalization_23_beta_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_163q
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:2
Identity_164╡
AssignVariableOp_164AssignVariableOp*assignvariableop_164_adam_dense_2_kernel_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_164q
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:2
Identity_165│
AssignVariableOp_165AssignVariableOp(assignvariableop_165_adam_dense_2_bias_vIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1659
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_166Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_166┌
Identity_167IdentityIdentity_166:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_167"%
identity_167Identity_167:output:0*у
_input_shapes╤
╬: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652*
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
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
п
▐
(__inference_model_2_layer_call_fn_435037

inputs"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:А&

unknown_13:АА

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А&

unknown_19:АА

unknown_20:	А%

unknown_21:А&

unknown_22:АА

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А%

unknown_28:А&

unknown_29:АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А&

unknown_35:АА

unknown_36:	А%

unknown_37:А&

unknown_38:А╪

unknown_39:	╪

unknown_40:	╪

unknown_41:	╪

unknown_42:	╪

unknown_43:	╪%

unknown_44:╪&

unknown_45:╪╪

unknown_46:	╪

unknown_47:	╪

unknown_48:	╪

unknown_49:	╪

unknown_50:	╪&

unknown_51:А╪

unknown_52:	╪%

unknown_53:╪&

unknown_54:╪А

unknown_55:	А

unknown_56:	А

unknown_57:	А

unknown_58:	А

unknown_59:	А

unknown_60:	А


unknown_61:

identityИвStatefulPartitionedCall│	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*Q
_read_only_resource_inputs3
1/	
 !"%&'()*+./01256789:;>?*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4335392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_431561

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_433109

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_22_layer_call_fn_436056

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4328752
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
З
m
A__inference_add_8_layer_call_and_return_conditional_losses_436081
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:         ╪2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         ╪:         ╪:Z V
0
_output_shapes
:         ╪
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:         ╪
"
_user_specified_name
inputs/1
Ў
╓
7__inference_batch_normalization_17_layer_call_fn_435285

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4310872
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
э
J
.__inference_activation_23_layer_call_fn_436221

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_4325852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_432470

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435651

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_430933

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_19_layer_call_fn_435584

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4314072
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Го
╗D
!__inference__wrapped_model_430911
input_3J
/model_2_conv2d_8_conv2d_readvariableop_resource:А?
0model_2_conv2d_8_biasadd_readvariableop_resource:	АE
6model_2_batch_normalization_16_readvariableop_resource:	АG
8model_2_batch_normalization_16_readvariableop_1_resource:	АV
Gmodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	АX
Imodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	А_
Dmodel_2_separable_conv2d_14_separable_conv2d_readvariableop_resource:Аb
Fmodel_2_separable_conv2d_14_separable_conv2d_readvariableop_1_resource:ААJ
;model_2_separable_conv2d_14_biasadd_readvariableop_resource:	АE
6model_2_batch_normalization_17_readvariableop_resource:	АG
8model_2_batch_normalization_17_readvariableop_1_resource:	АV
Gmodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	АX
Imodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	А_
Dmodel_2_separable_conv2d_15_separable_conv2d_readvariableop_resource:Аb
Fmodel_2_separable_conv2d_15_separable_conv2d_readvariableop_1_resource:ААJ
;model_2_separable_conv2d_15_biasadd_readvariableop_resource:	АE
6model_2_batch_normalization_18_readvariableop_resource:	АG
8model_2_batch_normalization_18_readvariableop_1_resource:	АV
Gmodel_2_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	АX
Imodel_2_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	АK
/model_2_conv2d_9_conv2d_readvariableop_resource:АА?
0model_2_conv2d_9_biasadd_readvariableop_resource:	А_
Dmodel_2_separable_conv2d_16_separable_conv2d_readvariableop_resource:Аb
Fmodel_2_separable_conv2d_16_separable_conv2d_readvariableop_1_resource:ААJ
;model_2_separable_conv2d_16_biasadd_readvariableop_resource:	АE
6model_2_batch_normalization_19_readvariableop_resource:	АG
8model_2_batch_normalization_19_readvariableop_1_resource:	АV
Gmodel_2_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	АX
Imodel_2_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	А_
Dmodel_2_separable_conv2d_17_separable_conv2d_readvariableop_resource:Аb
Fmodel_2_separable_conv2d_17_separable_conv2d_readvariableop_1_resource:ААJ
;model_2_separable_conv2d_17_biasadd_readvariableop_resource:	АE
6model_2_batch_normalization_20_readvariableop_resource:	АG
8model_2_batch_normalization_20_readvariableop_1_resource:	АV
Gmodel_2_batch_normalization_20_fusedbatchnormv3_readvariableop_resource:	АX
Imodel_2_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:	АL
0model_2_conv2d_10_conv2d_readvariableop_resource:АА@
1model_2_conv2d_10_biasadd_readvariableop_resource:	А_
Dmodel_2_separable_conv2d_18_separable_conv2d_readvariableop_resource:Аb
Fmodel_2_separable_conv2d_18_separable_conv2d_readvariableop_1_resource:А╪J
;model_2_separable_conv2d_18_biasadd_readvariableop_resource:	╪E
6model_2_batch_normalization_21_readvariableop_resource:	╪G
8model_2_batch_normalization_21_readvariableop_1_resource:	╪V
Gmodel_2_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	╪X
Imodel_2_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	╪_
Dmodel_2_separable_conv2d_19_separable_conv2d_readvariableop_resource:╪b
Fmodel_2_separable_conv2d_19_separable_conv2d_readvariableop_1_resource:╪╪J
;model_2_separable_conv2d_19_biasadd_readvariableop_resource:	╪E
6model_2_batch_normalization_22_readvariableop_resource:	╪G
8model_2_batch_normalization_22_readvariableop_1_resource:	╪V
Gmodel_2_batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	╪X
Imodel_2_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	╪L
0model_2_conv2d_11_conv2d_readvariableop_resource:А╪@
1model_2_conv2d_11_biasadd_readvariableop_resource:	╪_
Dmodel_2_separable_conv2d_20_separable_conv2d_readvariableop_resource:╪b
Fmodel_2_separable_conv2d_20_separable_conv2d_readvariableop_1_resource:╪АJ
;model_2_separable_conv2d_20_biasadd_readvariableop_resource:	АE
6model_2_batch_normalization_23_readvariableop_resource:	АG
8model_2_batch_normalization_23_readvariableop_1_resource:	АV
Gmodel_2_batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	АX
Imodel_2_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	АA
.model_2_dense_2_matmul_readvariableop_resource:	А
=
/model_2_dense_2_biasadd_readvariableop_resource:

identityИв>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_16/ReadVariableOpв/model_2/batch_normalization_16/ReadVariableOp_1в>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_17/ReadVariableOpв/model_2/batch_normalization_17/ReadVariableOp_1в>model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_18/ReadVariableOpв/model_2/batch_normalization_18/ReadVariableOp_1в>model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_19/ReadVariableOpв/model_2/batch_normalization_19/ReadVariableOp_1в>model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_20/ReadVariableOpв/model_2/batch_normalization_20/ReadVariableOp_1в>model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_21/ReadVariableOpв/model_2/batch_normalization_21/ReadVariableOp_1в>model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_22/ReadVariableOpв/model_2/batch_normalization_22/ReadVariableOp_1в>model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpв@model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1в-model_2/batch_normalization_23/ReadVariableOpв/model_2/batch_normalization_23/ReadVariableOp_1в(model_2/conv2d_10/BiasAdd/ReadVariableOpв'model_2/conv2d_10/Conv2D/ReadVariableOpв(model_2/conv2d_11/BiasAdd/ReadVariableOpв'model_2/conv2d_11/Conv2D/ReadVariableOpв'model_2/conv2d_8/BiasAdd/ReadVariableOpв&model_2/conv2d_8/Conv2D/ReadVariableOpв'model_2/conv2d_9/BiasAdd/ReadVariableOpв&model_2/conv2d_9/Conv2D/ReadVariableOpв&model_2/dense_2/BiasAdd/ReadVariableOpв%model_2/dense_2/MatMul/ReadVariableOpв2model_2/separable_conv2d_14/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_14/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1в2model_2/separable_conv2d_15/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_15/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1в2model_2/separable_conv2d_16/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_16/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1в2model_2/separable_conv2d_17/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_17/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1в2model_2/separable_conv2d_18/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_18/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_1в2model_2/separable_conv2d_19/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_19/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_1в2model_2/separable_conv2d_20/BiasAdd/ReadVariableOpв;model_2/separable_conv2d_20/separable_conv2d/ReadVariableOpв=model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1}
model_2/rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
model_2/rescaling_2/Cast/xБ
model_2/rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_2/rescaling_2/Cast_1/xб
model_2/rescaling_2/mulMulinput_3#model_2/rescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
model_2/rescaling_2/mul╣
model_2/rescaling_2/addAddV2model_2/rescaling_2/mul:z:0%model_2/rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
model_2/rescaling_2/add╔
&model_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02(
&model_2/conv2d_8/Conv2D/ReadVariableOpь
model_2/conv2d_8/Conv2DConv2Dmodel_2/rescaling_2/add:z:0.model_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
model_2/conv2d_8/Conv2D└
'model_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'model_2/conv2d_8/BiasAdd/ReadVariableOp═
model_2/conv2d_8/BiasAddBiasAdd model_2/conv2d_8/Conv2D:output:0/model_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
model_2/conv2d_8/BiasAdd╥
-model_2/batch_normalization_16/ReadVariableOpReadVariableOp6model_2_batch_normalization_16_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model_2/batch_normalization_16/ReadVariableOp╪
/model_2/batch_normalization_16/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/model_2/batch_normalization_16/ReadVariableOp_1Е
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1д
/model_2/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3!model_2/conv2d_8/BiasAdd:output:05model_2/batch_normalization_16/ReadVariableOp:value:07model_2/batch_normalization_16/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_16/FusedBatchNormV3░
model_2/activation_16/ReluRelu3model_2/batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:           А2
model_2/activation_16/Reluе
model_2/activation_17/ReluRelu(model_2/activation_16/Relu:activations:0*
T0*0
_output_shapes
:           А2
model_2/activation_17/ReluИ
;model_2/separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_14_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02=
;model_2/separable_conv2d_14/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_14_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02?
=model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      24
2model_2/separable_conv2d_14/separable_conv2d/Shape╔
:model_2/separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_14/separable_conv2d/dilation_rate█
6model_2/separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNative(model_2/activation_17/Relu:activations:0Cmodel_2/separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
28
6model_2/separable_conv2d_14/separable_conv2d/depthwise╥
,model_2/separable_conv2d_14/separable_conv2dConv2D?model_2/separable_conv2d_14/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2.
,model_2/separable_conv2d_14/separable_conv2dс
2model_2/separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype024
2model_2/separable_conv2d_14/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_14/BiasAddBiasAdd5model_2/separable_conv2d_14/separable_conv2d:output:0:model_2/separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2%
#model_2/separable_conv2d_14/BiasAdd╥
-model_2/batch_normalization_17/ReadVariableOpReadVariableOp6model_2_batch_normalization_17_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model_2/batch_normalization_17/ReadVariableOp╪
/model_2/batch_normalization_17/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/model_2/batch_normalization_17/ReadVariableOp_1Е
>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_14/BiasAdd:output:05model_2/batch_normalization_17/ReadVariableOp:value:07model_2/batch_normalization_17/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_17/FusedBatchNormV3░
model_2/activation_18/ReluRelu3model_2/batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:           А2
model_2/activation_18/ReluИ
;model_2/separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_15_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02=
;model_2/separable_conv2d_15/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_15_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02?
=model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            24
2model_2/separable_conv2d_15/separable_conv2d/Shape╔
:model_2/separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_15/separable_conv2d/dilation_rate█
6model_2/separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative(model_2/activation_18/Relu:activations:0Cmodel_2/separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
28
6model_2/separable_conv2d_15/separable_conv2d/depthwise╥
,model_2/separable_conv2d_15/separable_conv2dConv2D?model_2/separable_conv2d_15/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2.
,model_2/separable_conv2d_15/separable_conv2dс
2model_2/separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype024
2model_2/separable_conv2d_15/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_15/BiasAddBiasAdd5model_2/separable_conv2d_15/separable_conv2d:output:0:model_2/separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2%
#model_2/separable_conv2d_15/BiasAdd╥
-model_2/batch_normalization_18/ReadVariableOpReadVariableOp6model_2_batch_normalization_18_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model_2/batch_normalization_18/ReadVariableOp╪
/model_2/batch_normalization_18/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/model_2/batch_normalization_18/ReadVariableOp_1Е
>model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_15/BiasAdd:output:05model_2/batch_normalization_18/ReadVariableOp:value:07model_2/batch_normalization_18/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_18/FusedBatchNormV3я
model_2/max_pooling2d_6/MaxPoolMaxPool3model_2/batch_normalization_18/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2!
model_2/max_pooling2d_6/MaxPool╩
&model_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02(
&model_2/conv2d_9/Conv2D/ReadVariableOp∙
model_2/conv2d_9/Conv2DConv2D(model_2/activation_16/Relu:activations:0.model_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
model_2/conv2d_9/Conv2D└
'model_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'model_2/conv2d_9/BiasAdd/ReadVariableOp═
model_2/conv2d_9/BiasAddBiasAdd model_2/conv2d_9/Conv2D:output:0/model_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
model_2/conv2d_9/BiasAdd╖
model_2/add_6/addAddV2(model_2/max_pooling2d_6/MaxPool:output:0!model_2/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
model_2/add_6/addТ
model_2/activation_19/ReluRelumodel_2/add_6/add:z:0*
T0*0
_output_shapes
:         А2
model_2/activation_19/ReluИ
;model_2/separable_conv2d_16/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_16_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02=
;model_2/separable_conv2d_16/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_16_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02?
=model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_16/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            24
2model_2/separable_conv2d_16/separable_conv2d/Shape╔
:model_2/separable_conv2d_16/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_16/separable_conv2d/dilation_rate█
6model_2/separable_conv2d_16/separable_conv2d/depthwiseDepthwiseConv2dNative(model_2/activation_19/Relu:activations:0Cmodel_2/separable_conv2d_16/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
28
6model_2/separable_conv2d_16/separable_conv2d/depthwise╥
,model_2/separable_conv2d_16/separable_conv2dConv2D?model_2/separable_conv2d_16/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2.
,model_2/separable_conv2d_16/separable_conv2dс
2model_2/separable_conv2d_16/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype024
2model_2/separable_conv2d_16/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_16/BiasAddBiasAdd5model_2/separable_conv2d_16/separable_conv2d:output:0:model_2/separable_conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2%
#model_2/separable_conv2d_16/BiasAdd╥
-model_2/batch_normalization_19/ReadVariableOpReadVariableOp6model_2_batch_normalization_19_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model_2/batch_normalization_19/ReadVariableOp╪
/model_2/batch_normalization_19/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/model_2/batch_normalization_19/ReadVariableOp_1Е
>model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_16/BiasAdd:output:05model_2/batch_normalization_19/ReadVariableOp:value:07model_2/batch_normalization_19/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_19/FusedBatchNormV3░
model_2/activation_20/ReluRelu3model_2/batch_normalization_19/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
model_2/activation_20/ReluИ
;model_2/separable_conv2d_17/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_17_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02=
;model_2/separable_conv2d_17/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_17_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02?
=model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_17/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            24
2model_2/separable_conv2d_17/separable_conv2d/Shape╔
:model_2/separable_conv2d_17/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_17/separable_conv2d/dilation_rate█
6model_2/separable_conv2d_17/separable_conv2d/depthwiseDepthwiseConv2dNative(model_2/activation_20/Relu:activations:0Cmodel_2/separable_conv2d_17/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
28
6model_2/separable_conv2d_17/separable_conv2d/depthwise╥
,model_2/separable_conv2d_17/separable_conv2dConv2D?model_2/separable_conv2d_17/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2.
,model_2/separable_conv2d_17/separable_conv2dс
2model_2/separable_conv2d_17/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype024
2model_2/separable_conv2d_17/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_17/BiasAddBiasAdd5model_2/separable_conv2d_17/separable_conv2d:output:0:model_2/separable_conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2%
#model_2/separable_conv2d_17/BiasAdd╥
-model_2/batch_normalization_20/ReadVariableOpReadVariableOp6model_2_batch_normalization_20_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model_2/batch_normalization_20/ReadVariableOp╪
/model_2/batch_normalization_20/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_20_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/model_2/batch_normalization_20/ReadVariableOp_1Е
>model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_17/BiasAdd:output:05model_2/batch_normalization_20/ReadVariableOp:value:07model_2/batch_normalization_20/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_20/FusedBatchNormV3я
model_2/max_pooling2d_7/MaxPoolMaxPool3model_2/batch_normalization_20/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2!
model_2/max_pooling2d_7/MaxPool═
'model_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02)
'model_2/conv2d_10/Conv2D/ReadVariableOpщ
model_2/conv2d_10/Conv2DConv2Dmodel_2/add_6/add:z:0/model_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
model_2/conv2d_10/Conv2D├
(model_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(model_2/conv2d_10/BiasAdd/ReadVariableOp╤
model_2/conv2d_10/BiasAddBiasAdd!model_2/conv2d_10/Conv2D:output:00model_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
model_2/conv2d_10/BiasAdd╕
model_2/add_7/addAddV2(model_2/max_pooling2d_7/MaxPool:output:0"model_2/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
model_2/add_7/addТ
model_2/activation_21/ReluRelumodel_2/add_7/add:z:0*
T0*0
_output_shapes
:         А2
model_2/activation_21/ReluИ
;model_2/separable_conv2d_18/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_18_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02=
;model_2/separable_conv2d_18/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_18_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:А╪*
dtype02?
=model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_18/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            24
2model_2/separable_conv2d_18/separable_conv2d/Shape╔
:model_2/separable_conv2d_18/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_18/separable_conv2d/dilation_rate█
6model_2/separable_conv2d_18/separable_conv2d/depthwiseDepthwiseConv2dNative(model_2/activation_21/Relu:activations:0Cmodel_2/separable_conv2d_18/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
28
6model_2/separable_conv2d_18/separable_conv2d/depthwise╥
,model_2/separable_conv2d_18/separable_conv2dConv2D?model_2/separable_conv2d_18/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         ╪*
paddingVALID*
strides
2.
,model_2/separable_conv2d_18/separable_conv2dс
2model_2/separable_conv2d_18/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype024
2model_2/separable_conv2d_18/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_18/BiasAddBiasAdd5model_2/separable_conv2d_18/separable_conv2d:output:0:model_2/separable_conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2%
#model_2/separable_conv2d_18/BiasAdd╥
-model_2/batch_normalization_21/ReadVariableOpReadVariableOp6model_2_batch_normalization_21_readvariableop_resource*
_output_shapes	
:╪*
dtype02/
-model_2/batch_normalization_21/ReadVariableOp╪
/model_2/batch_normalization_21/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:╪*
dtype021
/model_2/batch_normalization_21/ReadVariableOp_1Е
>model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02@
>model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02B
@model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_18/BiasAdd:output:05model_2/batch_normalization_21/ReadVariableOp:value:07model_2/batch_normalization_21/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_21/FusedBatchNormV3░
model_2/activation_22/ReluRelu3model_2/batch_normalization_21/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╪2
model_2/activation_22/ReluИ
;model_2/separable_conv2d_19/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_19_separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype02=
;model_2/separable_conv2d_19/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_19_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪╪*
dtype02?
=model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_19/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     24
2model_2/separable_conv2d_19/separable_conv2d/Shape╔
:model_2/separable_conv2d_19/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_19/separable_conv2d/dilation_rate█
6model_2/separable_conv2d_19/separable_conv2d/depthwiseDepthwiseConv2dNative(model_2/activation_22/Relu:activations:0Cmodel_2/separable_conv2d_19/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
28
6model_2/separable_conv2d_19/separable_conv2d/depthwise╥
,model_2/separable_conv2d_19/separable_conv2dConv2D?model_2/separable_conv2d_19/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         ╪*
paddingVALID*
strides
2.
,model_2/separable_conv2d_19/separable_conv2dс
2model_2/separable_conv2d_19/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype024
2model_2/separable_conv2d_19/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_19/BiasAddBiasAdd5model_2/separable_conv2d_19/separable_conv2d:output:0:model_2/separable_conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2%
#model_2/separable_conv2d_19/BiasAdd╥
-model_2/batch_normalization_22/ReadVariableOpReadVariableOp6model_2_batch_normalization_22_readvariableop_resource*
_output_shapes	
:╪*
dtype02/
-model_2/batch_normalization_22/ReadVariableOp╪
/model_2/batch_normalization_22/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:╪*
dtype021
/model_2/batch_normalization_22/ReadVariableOp_1Е
>model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02@
>model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02B
@model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_19/BiasAdd:output:05model_2/batch_normalization_22/ReadVariableOp:value:07model_2/batch_normalization_22/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_22/FusedBatchNormV3я
model_2/max_pooling2d_8/MaxPoolMaxPool3model_2/batch_normalization_22/FusedBatchNormV3:y:0*0
_output_shapes
:         ╪*
ksize
*
paddingSAME*
strides
2!
model_2/max_pooling2d_8/MaxPool═
'model_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:А╪*
dtype02)
'model_2/conv2d_11/Conv2D/ReadVariableOpщ
model_2/conv2d_11/Conv2DConv2Dmodel_2/add_7/add:z:0/model_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
2
model_2/conv2d_11/Conv2D├
(model_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02*
(model_2/conv2d_11/BiasAdd/ReadVariableOp╤
model_2/conv2d_11/BiasAddBiasAdd!model_2/conv2d_11/Conv2D:output:00model_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
model_2/conv2d_11/BiasAdd╕
model_2/add_8/addAddV2(model_2/max_pooling2d_8/MaxPool:output:0"model_2/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         ╪2
model_2/add_8/addИ
;model_2/separable_conv2d_20/separable_conv2d/ReadVariableOpReadVariableOpDmodel_2_separable_conv2d_20_separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype02=
;model_2/separable_conv2d_20/separable_conv2d/ReadVariableOpП
=model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1ReadVariableOpFmodel_2_separable_conv2d_20_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪А*
dtype02?
=model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1┴
2model_2/separable_conv2d_20/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     24
2model_2/separable_conv2d_20/separable_conv2d/Shape╔
:model_2/separable_conv2d_20/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2<
:model_2/separable_conv2d_20/separable_conv2d/dilation_rate╚
6model_2/separable_conv2d_20/separable_conv2d/depthwiseDepthwiseConv2dNativemodel_2/add_8/add:z:0Cmodel_2/separable_conv2d_20/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
28
6model_2/separable_conv2d_20/separable_conv2d/depthwise╥
,model_2/separable_conv2d_20/separable_conv2dConv2D?model_2/separable_conv2d_20/separable_conv2d/depthwise:output:0Emodel_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2.
,model_2/separable_conv2d_20/separable_conv2dс
2model_2/separable_conv2d_20/BiasAdd/ReadVariableOpReadVariableOp;model_2_separable_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype024
2model_2/separable_conv2d_20/BiasAdd/ReadVariableOpГ
#model_2/separable_conv2d_20/BiasAddBiasAdd5model_2/separable_conv2d_20/separable_conv2d:output:0:model_2/separable_conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2%
#model_2/separable_conv2d_20/BiasAdd╥
-model_2/batch_normalization_23/ReadVariableOpReadVariableOp6model_2_batch_normalization_23_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model_2/batch_normalization_23/ReadVariableOp╪
/model_2/batch_normalization_23/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/model_2/batch_normalization_23/ReadVariableOp_1Е
>model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpЛ
@model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1п
/model_2/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3,model_2/separable_conv2d_20/BiasAdd:output:05model_2/batch_normalization_23/ReadVariableOp:value:07model_2/batch_normalization_23/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/model_2/batch_normalization_23/FusedBatchNormV3░
model_2/activation_23/ReluRelu3model_2/batch_normalization_23/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
model_2/activation_23/Relu╟
9model_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_2/global_average_pooling2d_2/Mean/reduction_indices√
'model_2/global_average_pooling2d_2/MeanMean(model_2/activation_23/Relu:activations:0Bmodel_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2)
'model_2/global_average_pooling2d_2/Meanй
model_2/dropout_2/IdentityIdentity0model_2/global_average_pooling2d_2/Mean:output:0*
T0*(
_output_shapes
:         А2
model_2/dropout_2/Identity╛
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp.model_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02'
%model_2/dense_2/MatMul/ReadVariableOp└
model_2/dense_2/MatMulMatMul#model_2/dropout_2/Identity:output:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
model_2/dense_2/MatMul╝
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&model_2/dense_2/BiasAdd/ReadVariableOp┴
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
model_2/dense_2/BiasAddС
model_2/dense_2/SoftmaxSoftmax model_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
model_2/dense_2/Softmaxл
IdentityIdentity!model_2/dense_2/Softmax:softmax:0?^model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_16/ReadVariableOp0^model_2/batch_normalization_16/ReadVariableOp_1?^model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_17/ReadVariableOp0^model_2/batch_normalization_17/ReadVariableOp_1?^model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_18/ReadVariableOp0^model_2/batch_normalization_18/ReadVariableOp_1?^model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_19/ReadVariableOp0^model_2/batch_normalization_19/ReadVariableOp_1?^model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_20/ReadVariableOp0^model_2/batch_normalization_20/ReadVariableOp_1?^model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_21/ReadVariableOp0^model_2/batch_normalization_21/ReadVariableOp_1?^model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_22/ReadVariableOp0^model_2/batch_normalization_22/ReadVariableOp_1?^model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_23/ReadVariableOp0^model_2/batch_normalization_23/ReadVariableOp_1)^model_2/conv2d_10/BiasAdd/ReadVariableOp(^model_2/conv2d_10/Conv2D/ReadVariableOp)^model_2/conv2d_11/BiasAdd/ReadVariableOp(^model_2/conv2d_11/Conv2D/ReadVariableOp(^model_2/conv2d_8/BiasAdd/ReadVariableOp'^model_2/conv2d_8/Conv2D/ReadVariableOp(^model_2/conv2d_9/BiasAdd/ReadVariableOp'^model_2/conv2d_9/Conv2D/ReadVariableOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp3^model_2/separable_conv2d_14/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_13^model_2/separable_conv2d_15/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_13^model_2/separable_conv2d_16/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_13^model_2/separable_conv2d_17/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_13^model_2/separable_conv2d_18/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_13^model_2/separable_conv2d_19/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_13^model_2/separable_conv2d_20/BiasAdd/ReadVariableOp<^model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp>^model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2А
>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_16/ReadVariableOp-model_2/batch_normalization_16/ReadVariableOp2b
/model_2/batch_normalization_16/ReadVariableOp_1/model_2/batch_normalization_16/ReadVariableOp_12А
>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_17/ReadVariableOp-model_2/batch_normalization_17/ReadVariableOp2b
/model_2/batch_normalization_17/ReadVariableOp_1/model_2/batch_normalization_17/ReadVariableOp_12А
>model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_18/ReadVariableOp-model_2/batch_normalization_18/ReadVariableOp2b
/model_2/batch_normalization_18/ReadVariableOp_1/model_2/batch_normalization_18/ReadVariableOp_12А
>model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_19/ReadVariableOp-model_2/batch_normalization_19/ReadVariableOp2b
/model_2/batch_normalization_19/ReadVariableOp_1/model_2/batch_normalization_19/ReadVariableOp_12А
>model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_20/ReadVariableOp-model_2/batch_normalization_20/ReadVariableOp2b
/model_2/batch_normalization_20/ReadVariableOp_1/model_2/batch_normalization_20/ReadVariableOp_12А
>model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_21/ReadVariableOp-model_2/batch_normalization_21/ReadVariableOp2b
/model_2/batch_normalization_21/ReadVariableOp_1/model_2/batch_normalization_21/ReadVariableOp_12А
>model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_22/ReadVariableOp-model_2/batch_normalization_22/ReadVariableOp2b
/model_2/batch_normalization_22/ReadVariableOp_1/model_2/batch_normalization_22/ReadVariableOp_12А
>model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2Д
@model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_23/ReadVariableOp-model_2/batch_normalization_23/ReadVariableOp2b
/model_2/batch_normalization_23/ReadVariableOp_1/model_2/batch_normalization_23/ReadVariableOp_12T
(model_2/conv2d_10/BiasAdd/ReadVariableOp(model_2/conv2d_10/BiasAdd/ReadVariableOp2R
'model_2/conv2d_10/Conv2D/ReadVariableOp'model_2/conv2d_10/Conv2D/ReadVariableOp2T
(model_2/conv2d_11/BiasAdd/ReadVariableOp(model_2/conv2d_11/BiasAdd/ReadVariableOp2R
'model_2/conv2d_11/Conv2D/ReadVariableOp'model_2/conv2d_11/Conv2D/ReadVariableOp2R
'model_2/conv2d_8/BiasAdd/ReadVariableOp'model_2/conv2d_8/BiasAdd/ReadVariableOp2P
&model_2/conv2d_8/Conv2D/ReadVariableOp&model_2/conv2d_8/Conv2D/ReadVariableOp2R
'model_2/conv2d_9/BiasAdd/ReadVariableOp'model_2/conv2d_9/BiasAdd/ReadVariableOp2P
&model_2/conv2d_9/Conv2D/ReadVariableOp&model_2/conv2d_9/Conv2D/ReadVariableOp2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp2h
2model_2/separable_conv2d_14/BiasAdd/ReadVariableOp2model_2/separable_conv2d_14/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_12h
2model_2/separable_conv2d_15/BiasAdd/ReadVariableOp2model_2/separable_conv2d_15/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_12h
2model_2/separable_conv2d_16/BiasAdd/ReadVariableOp2model_2/separable_conv2d_16/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_12h
2model_2/separable_conv2d_17/BiasAdd/ReadVariableOp2model_2/separable_conv2d_17/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_12h
2model_2/separable_conv2d_18/BiasAdd/ReadVariableOp2model_2/separable_conv2d_18/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_18/separable_conv2d/ReadVariableOp_12h
2model_2/separable_conv2d_19/BiasAdd/ReadVariableOp2model_2/separable_conv2d_19/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_19/separable_conv2d/ReadVariableOp_12h
2model_2/separable_conv2d_20/BiasAdd/ReadVariableOp2model_2/separable_conv2d_20/BiasAdd/ReadVariableOp2z
;model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp;model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp2~
=model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1=model_2/separable_conv2d_20/separable_conv2d/ReadVariableOp_1:X T
/
_output_shapes
:         @@
!
_user_specified_name	input_3
э
J
.__inference_activation_17_layer_call_fn_435200

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_4322302
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_431131

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_432297

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435074

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_431771

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_20_layer_call_fn_435731

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4316052
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_431407

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
▓
▀
(__inference_model_2_layer_call_fn_433799
input_3"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:А&

unknown_13:АА

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А&

unknown_19:АА

unknown_20:	А%

unknown_21:А&

unknown_22:АА

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А%

unknown_28:А&

unknown_29:АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А&

unknown_35:АА

unknown_36:	А%

unknown_37:А&

unknown_38:А╪

unknown_39:	╪

unknown_40:	╪

unknown_41:	╪

unknown_42:	╪

unknown_43:	╪%

unknown_44:╪&

unknown_45:╪╪

unknown_46:	╪

unknown_47:	╪

unknown_48:	╪

unknown_49:	╪

unknown_50:	╪&

unknown_51:А╪

unknown_52:	╪%

unknown_53:╪&

unknown_54:╪А

unknown_55:	А

unknown_56:	А

unknown_57:	А

unknown_58:	А

unknown_59:	А

unknown_60:	А


unknown_61:

identityИвStatefulPartitionedCall┤	
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*Q
_read_only_resource_inputs3
1/	
 !"%&'()*+./01256789:;>?*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4335392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         @@
!
_user_specified_name	input_3
Ї
R
&__inference_add_7_layer_call_fn_435788
inputs_0
inputs_1
identity╪
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_4324372
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:Z V
0
_output_shapes
:         А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:         А
"
_user_specified_name
inputs/1
к
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_431351

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
 
k
A__inference_add_8_layer_call_and_return_conditional_losses_432544

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:         ╪2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         ╪:         ╪:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs:XT
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435950

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
Е
e
I__inference_activation_19_layer_call_and_return_conditional_losses_435494

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436159

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
 
k
A__inference_add_6_layer_call_and_return_conditional_losses_432330

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:         А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs:XT
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435705

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_431881

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
┐
▐
(__inference_model_2_layer_call_fn_434906

inputs"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:А&

unknown_13:АА

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А&

unknown_19:АА

unknown_20:	А%

unknown_21:А&

unknown_22:АА

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А%

unknown_28:А&

unknown_29:АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А&

unknown_35:АА

unknown_36:	А%

unknown_37:А&

unknown_38:А╪

unknown_39:	╪

unknown_40:	╪

unknown_41:	╪

unknown_42:	╪

unknown_43:	╪%

unknown_44:╪&

unknown_45:╪╪

unknown_46:	╪

unknown_47:	╪

unknown_48:	╪

unknown_49:	╪

unknown_50:	╪&

unknown_51:А╪

unknown_52:	╪%

unknown_53:╪&

unknown_54:╪А

unknown_55:	А

unknown_56:	А

unknown_57:	А

unknown_58:	А

unknown_59:	А

unknown_60:	А


unknown_61:

identityИвStatefulPartitionedCall├	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4326132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Е
e
I__inference_activation_22_layer_call_and_return_conditional_losses_435927

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╪2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╪:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_22_layer_call_fn_436043

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4325112
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
═
б
)__inference_conv2d_9_layer_call_fn_435477

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_4323182
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :           А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
┌
L
0__inference_max_pooling2d_8_layer_call_fn_431997

inputs
identityя
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
GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_4319912
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_19_layer_call_fn_435597

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4314512
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435110

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435370

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Е
e
I__inference_activation_16_layer_call_and_return_conditional_losses_435185

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:           А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_432047

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
э
J
.__inference_activation_19_layer_call_fn_435499

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_4323372
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_431207

inputsC
(separable_conv2d_readvariableop_resource:АF
*separable_conv2d_readvariableop_1_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ў
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_436226

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435669

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╝

Б
E__inference_conv2d_11_layer_call_and_return_conditional_losses_432532

inputs:
conv2d_readvariableop_resource:А╪.
biasadd_readvariableop_resource:	╪
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:А╪*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435254

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_431605

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435388

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435687

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_21_layer_call_fn_435909

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4324702
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_23_layer_call_fn_436198

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4325702
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435406

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
┬
▀
(__inference_model_2_layer_call_fn_432742
input_3"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А%
	unknown_6:АА
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А%

unknown_12:А&

unknown_13:АА

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А&

unknown_19:АА

unknown_20:	А%

unknown_21:А&

unknown_22:АА

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А%

unknown_28:А&

unknown_29:АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А&

unknown_35:АА

unknown_36:	А%

unknown_37:А&

unknown_38:А╪

unknown_39:	╪

unknown_40:	╪

unknown_41:	╪

unknown_42:	╪

unknown_43:	╪%

unknown_44:╪&

unknown_45:╪╪

unknown_46:	╪

unknown_47:	╪

unknown_48:	╪

unknown_49:	╪

unknown_50:	╪&

unknown_51:А╪

unknown_52:	╪%

unknown_53:╪&

unknown_54:╪А

unknown_55:	А

unknown_56:	А

unknown_57:	А

unknown_58:	А

unknown_59:	А

unknown_60:	А


unknown_61:

identityИвStatefulPartitionedCall─	
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*a
_read_only_resource_inputsC
A?	
 !"#$%&'()*+,-./0123456789:;<=>?*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4326132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         @@
!
_user_specified_name	input_3
р
╥
4__inference_separable_conv2d_15_layer_call_fn_431219

inputs"
unknown:А%
	unknown_0:АА
	unknown_1:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_4312072
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_21_layer_call_fn_435896

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4317712
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_17_layer_call_fn_435311

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4322562
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
╗

А
D__inference_conv2d_9_layer_call_and_return_conditional_losses_432318

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :           А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
р
╥
4__inference_separable_conv2d_14_layer_call_fn_431065

inputs"
unknown:А%
	unknown_0:АА
	unknown_1:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_4310532
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_19_layer_call_fn_435623

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4330422
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
о
╓
7__inference_batch_normalization_20_layer_call_fn_435744

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4324042
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_432404

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Е
e
I__inference_activation_21_layer_call_and_return_conditional_losses_432444

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435128

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
р
╥
4__inference_separable_conv2d_16_layer_call_fn_431385

inputs"
unknown:А%
	unknown_0:АА
	unknown_1:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_4313732
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_432992

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_18_layer_call_fn_435458

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4331092
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435352

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Е
e
I__inference_activation_18_layer_call_and_return_conditional_losses_432271

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:           А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:           А:X T
0
_output_shapes
:           А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435571

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
м
╓
7__inference_batch_normalization_20_layer_call_fn_435757

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4329922
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
┼
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435870

inputs&
readvariableop_resource:	╪(
readvariableop_1_resource:	╪7
(fusedbatchnormv3_readvariableop_resource:	╪9
*fusedbatchnormv3_readvariableop_1_resource:	╪
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:╪*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         ╪: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ╪
 
_user_specified_nameinputs
╖

 
D__inference_conv2d_8_layer_call_and_return_conditional_losses_432185

inputs9
conv2d_readvariableop_resource:А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
З
m
A__inference_add_6_layer_call_and_return_conditional_losses_435483
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:         А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:Z V
0
_output_shapes
:         А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:         А
"
_user_specified_name
inputs/1
Е
e
I__inference_activation_20_layer_call_and_return_conditional_losses_435628

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ж¤
ь<
C__inference_model_2_layer_call_and_return_conditional_losses_434523

inputsB
'conv2d_8_conv2d_readvariableop_resource:А7
(conv2d_8_biasadd_readvariableop_resource:	А=
.batch_normalization_16_readvariableop_resource:	А?
0batch_normalization_16_readvariableop_1_resource:	АN
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	АW
<separable_conv2d_14_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_14_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_14_biasadd_readvariableop_resource:	А=
.batch_normalization_17_readvariableop_resource:	А?
0batch_normalization_17_readvariableop_1_resource:	АN
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	АW
<separable_conv2d_15_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_15_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_15_biasadd_readvariableop_resource:	А=
.batch_normalization_18_readvariableop_resource:	А?
0batch_normalization_18_readvariableop_1_resource:	АN
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	АC
'conv2d_9_conv2d_readvariableop_resource:АА7
(conv2d_9_biasadd_readvariableop_resource:	АW
<separable_conv2d_16_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_16_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_16_biasadd_readvariableop_resource:	А=
.batch_normalization_19_readvariableop_resource:	А?
0batch_normalization_19_readvariableop_1_resource:	АN
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	АW
<separable_conv2d_17_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_17_separable_conv2d_readvariableop_1_resource:ААB
3separable_conv2d_17_biasadd_readvariableop_resource:	А=
.batch_normalization_20_readvariableop_resource:	А?
0batch_normalization_20_readvariableop_1_resource:	АN
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:	АD
(conv2d_10_conv2d_readvariableop_resource:АА8
)conv2d_10_biasadd_readvariableop_resource:	АW
<separable_conv2d_18_separable_conv2d_readvariableop_resource:АZ
>separable_conv2d_18_separable_conv2d_readvariableop_1_resource:А╪B
3separable_conv2d_18_biasadd_readvariableop_resource:	╪=
.batch_normalization_21_readvariableop_resource:	╪?
0batch_normalization_21_readvariableop_1_resource:	╪N
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	╪P
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	╪W
<separable_conv2d_19_separable_conv2d_readvariableop_resource:╪Z
>separable_conv2d_19_separable_conv2d_readvariableop_1_resource:╪╪B
3separable_conv2d_19_biasadd_readvariableop_resource:	╪=
.batch_normalization_22_readvariableop_resource:	╪?
0batch_normalization_22_readvariableop_1_resource:	╪N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	╪P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	╪D
(conv2d_11_conv2d_readvariableop_resource:А╪8
)conv2d_11_biasadd_readvariableop_resource:	╪W
<separable_conv2d_20_separable_conv2d_readvariableop_resource:╪Z
>separable_conv2d_20_separable_conv2d_readvariableop_1_resource:╪АB
3separable_conv2d_20_biasadd_readvariableop_resource:	А=
.batch_normalization_23_readvariableop_resource:	А?
0batch_normalization_23_readvariableop_1_resource:	АN
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	А9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:

identityИв6batch_normalization_16/FusedBatchNormV3/ReadVariableOpв8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_16/ReadVariableOpв'batch_normalization_16/ReadVariableOp_1в6batch_normalization_17/FusedBatchNormV3/ReadVariableOpв8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_17/ReadVariableOpв'batch_normalization_17/ReadVariableOp_1в6batch_normalization_18/FusedBatchNormV3/ReadVariableOpв8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_18/ReadVariableOpв'batch_normalization_18/ReadVariableOp_1в6batch_normalization_19/FusedBatchNormV3/ReadVariableOpв8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_19/ReadVariableOpв'batch_normalization_19/ReadVariableOp_1в6batch_normalization_20/FusedBatchNormV3/ReadVariableOpв8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_20/ReadVariableOpв'batch_normalization_20/ReadVariableOp_1в6batch_normalization_21/FusedBatchNormV3/ReadVariableOpв8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_21/ReadVariableOpв'batch_normalization_21/ReadVariableOp_1в6batch_normalization_22/FusedBatchNormV3/ReadVariableOpв8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_22/ReadVariableOpв'batch_normalization_22/ReadVariableOp_1в6batch_normalization_23/FusedBatchNormV3/ReadVariableOpв8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_23/ReadVariableOpв'batch_normalization_23/ReadVariableOp_1в conv2d_10/BiasAdd/ReadVariableOpвconv2d_10/Conv2D/ReadVariableOpв conv2d_11/BiasAdd/ReadVariableOpвconv2d_11/Conv2D/ReadVariableOpвconv2d_8/BiasAdd/ReadVariableOpвconv2d_8/Conv2D/ReadVariableOpвconv2d_9/BiasAdd/ReadVariableOpвconv2d_9/Conv2D/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpв*separable_conv2d_14/BiasAdd/ReadVariableOpв3separable_conv2d_14/separable_conv2d/ReadVariableOpв5separable_conv2d_14/separable_conv2d/ReadVariableOp_1в*separable_conv2d_15/BiasAdd/ReadVariableOpв3separable_conv2d_15/separable_conv2d/ReadVariableOpв5separable_conv2d_15/separable_conv2d/ReadVariableOp_1в*separable_conv2d_16/BiasAdd/ReadVariableOpв3separable_conv2d_16/separable_conv2d/ReadVariableOpв5separable_conv2d_16/separable_conv2d/ReadVariableOp_1в*separable_conv2d_17/BiasAdd/ReadVariableOpв3separable_conv2d_17/separable_conv2d/ReadVariableOpв5separable_conv2d_17/separable_conv2d/ReadVariableOp_1в*separable_conv2d_18/BiasAdd/ReadVariableOpв3separable_conv2d_18/separable_conv2d/ReadVariableOpв5separable_conv2d_18/separable_conv2d/ReadVariableOp_1в*separable_conv2d_19/BiasAdd/ReadVariableOpв3separable_conv2d_19/separable_conv2d/ReadVariableOpв5separable_conv2d_19/separable_conv2d/ReadVariableOp_1в*separable_conv2d_20/BiasAdd/ReadVariableOpв3separable_conv2d_20/separable_conv2d/ReadVariableOpв5separable_conv2d_20/separable_conv2d/ReadVariableOp_1m
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_2/Cast/xq
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_2/Cast_1/xИ
rescaling_2/mulMulinputsrescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/mulЩ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/add▒
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02 
conv2d_8/Conv2D/ReadVariableOp╠
conv2d_8/Conv2DConv2Drescaling_2/add:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
conv2d_8/Conv2Dи
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpн
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
conv2d_8/BiasAdd║
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_16/ReadVariableOp└
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_16/ReadVariableOp_1э
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ь
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3Ш
activation_16/ReluRelu+batch_normalization_16/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:           А2
activation_16/ReluН
activation_17/ReluRelu activation_16/Relu:activations:0*
T0*0
_output_shapes
:           А2
activation_17/ReluЁ
3separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_14_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_14/separable_conv2d/ReadVariableOpў
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_14_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2,
*separable_conv2d_14/separable_conv2d/Shape╣
2separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_14/separable_conv2d/dilation_rate╗
.separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNative activation_17/Relu:activations:0;separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_14/separable_conv2d/depthwise▓
$separable_conv2d_14/separable_conv2dConv2D7separable_conv2d_14/separable_conv2d/depthwise:output:0=separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_14/separable_conv2d╔
*separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_14/BiasAdd/ReadVariableOpу
separable_conv2d_14/BiasAddBiasAdd-separable_conv2d_14/separable_conv2d:output:02separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_14/BiasAdd║
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_17/ReadVariableOp└
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_17/ReadVariableOp_1э
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_14/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3Ш
activation_18/ReluRelu+batch_normalization_17/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:           А2
activation_18/ReluЁ
3separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_15_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_15/separable_conv2d/ReadVariableOpў
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_15_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_15/separable_conv2d/Shape╣
2separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_15/separable_conv2d/dilation_rate╗
.separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative activation_18/Relu:activations:0;separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
20
.separable_conv2d_15/separable_conv2d/depthwise▓
$separable_conv2d_15/separable_conv2dConv2D7separable_conv2d_15/separable_conv2d/depthwise:output:0=separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:           А*
paddingVALID*
strides
2&
$separable_conv2d_15/separable_conv2d╔
*separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_15/BiasAdd/ReadVariableOpу
separable_conv2d_15/BiasAddBiasAdd-separable_conv2d_15/separable_conv2d:output:02separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2
separable_conv2d_15/BiasAdd║
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_18/ReadVariableOp└
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_18/ReadVariableOp_1э
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_15/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3╫
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_18/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d_6/MaxPool▓
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_9/Conv2D/ReadVariableOp┘
conv2d_9/Conv2DConv2D activation_16/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_9/Conv2Dи
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_9/BiasAdd/ReadVariableOpн
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_9/BiasAddЧ
	add_6/addAddV2 max_pooling2d_6/MaxPool:output:0conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
	add_6/addz
activation_19/ReluReluadd_6/add:z:0*
T0*0
_output_shapes
:         А2
activation_19/ReluЁ
3separable_conv2d_16/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_16_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_16/separable_conv2d/ReadVariableOpў
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_16_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_16/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_16/separable_conv2d/Shape╣
2separable_conv2d_16/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_16/separable_conv2d/dilation_rate╗
.separable_conv2d_16/separable_conv2d/depthwiseDepthwiseConv2dNative activation_19/Relu:activations:0;separable_conv2d_16/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
20
.separable_conv2d_16/separable_conv2d/depthwise▓
$separable_conv2d_16/separable_conv2dConv2D7separable_conv2d_16/separable_conv2d/depthwise:output:0=separable_conv2d_16/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2&
$separable_conv2d_16/separable_conv2d╔
*separable_conv2d_16/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_16/BiasAdd/ReadVariableOpу
separable_conv2d_16/BiasAddBiasAdd-separable_conv2d_16/separable_conv2d:output:02separable_conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
separable_conv2d_16/BiasAdd║
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_19/ReadVariableOp└
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_19/ReadVariableOp_1э
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_16/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3Ш
activation_20/ReluRelu+batch_normalization_19/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
activation_20/ReluЁ
3separable_conv2d_17/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_17_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_17/separable_conv2d/ReadVariableOpў
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_17_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype027
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_17/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_17/separable_conv2d/Shape╣
2separable_conv2d_17/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_17/separable_conv2d/dilation_rate╗
.separable_conv2d_17/separable_conv2d/depthwiseDepthwiseConv2dNative activation_20/Relu:activations:0;separable_conv2d_17/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
20
.separable_conv2d_17/separable_conv2d/depthwise▓
$separable_conv2d_17/separable_conv2dConv2D7separable_conv2d_17/separable_conv2d/depthwise:output:0=separable_conv2d_17/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2&
$separable_conv2d_17/separable_conv2d╔
*separable_conv2d_17/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_17/BiasAdd/ReadVariableOpу
separable_conv2d_17/BiasAddBiasAdd-separable_conv2d_17/separable_conv2d:output:02separable_conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
separable_conv2d_17/BiasAdd║
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_20/ReadVariableOp└
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_20/ReadVariableOp_1э
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_17/BiasAdd:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_20/FusedBatchNormV3╫
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_20/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d_7/MaxPool╡
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_10/Conv2D/ReadVariableOp╔
conv2d_10/Conv2DConv2Dadd_6/add:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_10/Conv2Dл
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▒
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_10/BiasAddШ
	add_7/addAddV2 max_pooling2d_7/MaxPool:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
	add_7/addz
activation_21/ReluReluadd_7/add:z:0*
T0*0
_output_shapes
:         А2
activation_21/ReluЁ
3separable_conv2d_18/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_18_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype025
3separable_conv2d_18/separable_conv2d/ReadVariableOpў
5separable_conv2d_18/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_18_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:А╪*
dtype027
5separable_conv2d_18/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_18/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_18/separable_conv2d/Shape╣
2separable_conv2d_18/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_18/separable_conv2d/dilation_rate╗
.separable_conv2d_18/separable_conv2d/depthwiseDepthwiseConv2dNative activation_21/Relu:activations:0;separable_conv2d_18/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
20
.separable_conv2d_18/separable_conv2d/depthwise▓
$separable_conv2d_18/separable_conv2dConv2D7separable_conv2d_18/separable_conv2d/depthwise:output:0=separable_conv2d_18/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         ╪*
paddingVALID*
strides
2&
$separable_conv2d_18/separable_conv2d╔
*separable_conv2d_18/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02,
*separable_conv2d_18/BiasAdd/ReadVariableOpу
separable_conv2d_18/BiasAddBiasAdd-separable_conv2d_18/separable_conv2d:output:02separable_conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
separable_conv2d_18/BiasAdd║
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:╪*
dtype02'
%batch_normalization_21/ReadVariableOp└
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02)
'batch_normalization_21/ReadVariableOp_1э
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_18/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_21/FusedBatchNormV3Ш
activation_22/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╪2
activation_22/ReluЁ
3separable_conv2d_19/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_19_separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype025
3separable_conv2d_19/separable_conv2d/ReadVariableOpў
5separable_conv2d_19/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_19_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪╪*
dtype027
5separable_conv2d_19/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_19/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     2,
*separable_conv2d_19/separable_conv2d/Shape╣
2separable_conv2d_19/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_19/separable_conv2d/dilation_rate╗
.separable_conv2d_19/separable_conv2d/depthwiseDepthwiseConv2dNative activation_22/Relu:activations:0;separable_conv2d_19/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
20
.separable_conv2d_19/separable_conv2d/depthwise▓
$separable_conv2d_19/separable_conv2dConv2D7separable_conv2d_19/separable_conv2d/depthwise:output:0=separable_conv2d_19/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         ╪*
paddingVALID*
strides
2&
$separable_conv2d_19/separable_conv2d╔
*separable_conv2d_19/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02,
*separable_conv2d_19/BiasAdd/ReadVariableOpу
separable_conv2d_19/BiasAddBiasAdd-separable_conv2d_19/separable_conv2d:output:02separable_conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
separable_conv2d_19/BiasAdd║
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:╪*
dtype02'
%batch_normalization_22/ReadVariableOp└
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02)
'batch_normalization_22/ReadVariableOp_1э
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:╪*
dtype028
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:╪*
dtype02:
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_19/BiasAdd:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ╪:╪:╪:╪:╪:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_22/FusedBatchNormV3╫
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_22/FusedBatchNormV3:y:0*0
_output_shapes
:         ╪*
ksize
*
paddingSAME*
strides
2
max_pooling2d_8/MaxPool╡
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:А╪*
dtype02!
conv2d_11/Conv2D/ReadVariableOp╔
conv2d_11/Conv2DConv2Dadd_7/add:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
2
conv2d_11/Conv2Dл
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp▒
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2
conv2d_11/BiasAddШ
	add_8/addAddV2 max_pooling2d_8/MaxPool:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:         ╪2
	add_8/addЁ
3separable_conv2d_20/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_20_separable_conv2d_readvariableop_resource*'
_output_shapes
:╪*
dtype025
3separable_conv2d_20/separable_conv2d/ReadVariableOpў
5separable_conv2d_20/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_20_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:╪А*
dtype027
5separable_conv2d_20/separable_conv2d/ReadVariableOp_1▒
*separable_conv2d_20/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      ╪     2,
*separable_conv2d_20/separable_conv2d/Shape╣
2separable_conv2d_20/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_20/separable_conv2d/dilation_rateи
.separable_conv2d_20/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_8/add:z:0;separable_conv2d_20/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
20
.separable_conv2d_20/separable_conv2d/depthwise▓
$separable_conv2d_20/separable_conv2dConv2D7separable_conv2d_20/separable_conv2d/depthwise:output:0=separable_conv2d_20/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2&
$separable_conv2d_20/separable_conv2d╔
*separable_conv2d_20/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*separable_conv2d_20/BiasAdd/ReadVariableOpу
separable_conv2d_20/BiasAddBiasAdd-separable_conv2d_20/separable_conv2d:output:02separable_conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
separable_conv2d_20/BiasAdd║
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_23/ReadVariableOp└
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_23/ReadVariableOp_1э
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ў
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3$separable_conv2d_20/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_23/FusedBatchNormV3Ш
activation_23/ReluRelu+batch_normalization_23/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
activation_23/Relu╖
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_2/Mean/reduction_indices█
global_average_pooling2d_2/MeanMean activation_23/Relu:activations:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2!
global_average_pooling2d_2/MeanС
dropout_2/IdentityIdentity(global_average_pooling2d_2/Mean:output:0*
T0*(
_output_shapes
:         А2
dropout_2/Identityж
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_2/MatMul/ReadVariableOpа
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_2/Softmaxл
IdentityIdentitydense_2/Softmax:softmax:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp+^separable_conv2d_14/BiasAdd/ReadVariableOp4^separable_conv2d_14/separable_conv2d/ReadVariableOp6^separable_conv2d_14/separable_conv2d/ReadVariableOp_1+^separable_conv2d_15/BiasAdd/ReadVariableOp4^separable_conv2d_15/separable_conv2d/ReadVariableOp6^separable_conv2d_15/separable_conv2d/ReadVariableOp_1+^separable_conv2d_16/BiasAdd/ReadVariableOp4^separable_conv2d_16/separable_conv2d/ReadVariableOp6^separable_conv2d_16/separable_conv2d/ReadVariableOp_1+^separable_conv2d_17/BiasAdd/ReadVariableOp4^separable_conv2d_17/separable_conv2d/ReadVariableOp6^separable_conv2d_17/separable_conv2d/ReadVariableOp_1+^separable_conv2d_18/BiasAdd/ReadVariableOp4^separable_conv2d_18/separable_conv2d/ReadVariableOp6^separable_conv2d_18/separable_conv2d/ReadVariableOp_1+^separable_conv2d_19/BiasAdd/ReadVariableOp4^separable_conv2d_19/separable_conv2d/ReadVariableOp6^separable_conv2d_19/separable_conv2d/ReadVariableOp_1+^separable_conv2d_20/BiasAdd/ReadVariableOp4^separable_conv2d_20/separable_conv2d/ReadVariableOp6^separable_conv2d_20/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2X
*separable_conv2d_14/BiasAdd/ReadVariableOp*separable_conv2d_14/BiasAdd/ReadVariableOp2j
3separable_conv2d_14/separable_conv2d/ReadVariableOp3separable_conv2d_14/separable_conv2d/ReadVariableOp2n
5separable_conv2d_14/separable_conv2d/ReadVariableOp_15separable_conv2d_14/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_15/BiasAdd/ReadVariableOp*separable_conv2d_15/BiasAdd/ReadVariableOp2j
3separable_conv2d_15/separable_conv2d/ReadVariableOp3separable_conv2d_15/separable_conv2d/ReadVariableOp2n
5separable_conv2d_15/separable_conv2d/ReadVariableOp_15separable_conv2d_15/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_16/BiasAdd/ReadVariableOp*separable_conv2d_16/BiasAdd/ReadVariableOp2j
3separable_conv2d_16/separable_conv2d/ReadVariableOp3separable_conv2d_16/separable_conv2d/ReadVariableOp2n
5separable_conv2d_16/separable_conv2d/ReadVariableOp_15separable_conv2d_16/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_17/BiasAdd/ReadVariableOp*separable_conv2d_17/BiasAdd/ReadVariableOp2j
3separable_conv2d_17/separable_conv2d/ReadVariableOp3separable_conv2d_17/separable_conv2d/ReadVariableOp2n
5separable_conv2d_17/separable_conv2d/ReadVariableOp_15separable_conv2d_17/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_18/BiasAdd/ReadVariableOp*separable_conv2d_18/BiasAdd/ReadVariableOp2j
3separable_conv2d_18/separable_conv2d/ReadVariableOp3separable_conv2d_18/separable_conv2d/ReadVariableOp2n
5separable_conv2d_18/separable_conv2d/ReadVariableOp_15separable_conv2d_18/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_19/BiasAdd/ReadVariableOp*separable_conv2d_19/BiasAdd/ReadVariableOp2j
3separable_conv2d_19/separable_conv2d/ReadVariableOp3separable_conv2d_19/separable_conv2d/ReadVariableOp2n
5separable_conv2d_19/separable_conv2d/ReadVariableOp_15separable_conv2d_19/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_20/BiasAdd/ReadVariableOp*separable_conv2d_20/BiasAdd/ReadVariableOp2j
3separable_conv2d_20/separable_conv2d/ReadVariableOp3separable_conv2d_20/separable_conv2d/ReadVariableOp2n
5separable_conv2d_20/separable_conv2d/ReadVariableOp_15separable_conv2d_20/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_18_layer_call_fn_435419

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4312412
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╧
в
*__inference_conv2d_10_layer_call_fn_435776

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_4324252
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_431373

inputsC
(separable_conv2d_readvariableop_resource:АF
*separable_conv2d_readvariableop_1_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ш
К
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_431693

inputsC
(separable_conv2d_readvariableop_resource:АF
*separable_conv2d_readvariableop_1_resource:А╪.
biasadd_readvariableop_resource:	╪
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1┤
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOp╗
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:А╪*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateў
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
separable_conv2d/depthwiseЇ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           ╪*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpе
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ╪2	
BiasAdd▐
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           А: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
к
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_431671

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_17_layer_call_fn_435298

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4311312
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ц
r
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_432158

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╝

Б
E__inference_conv2d_11_layer_call_and_return_conditional_losses_436066

inputs:
conv2d_readvariableop_resource:А╪.
biasadd_readvariableop_resource:	╪
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:А╪*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╪2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╖

 
D__inference_conv2d_8_layer_call_and_return_conditional_losses_435047

inputs9
conv2d_readvariableop_resource:А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╤╫
йS
__inference__traced_save_436789
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableopC
?savev2_separable_conv2d_14_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_14_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableopC
?savev2_separable_conv2d_15_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_15_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableopC
?savev2_separable_conv2d_16_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_16_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableopC
?savev2_separable_conv2d_17_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_17_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableopC
?savev2_separable_conv2d_18_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_18_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableopC
?savev2_separable_conv2d_19_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_19_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_19_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableopC
?savev2_separable_conv2d_20_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_20_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_20_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_14_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_14_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_14_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_15_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_15_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_15_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_16_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_16_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_16_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_17_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_17_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_18_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_18_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_19_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_19_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_19_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_20_depthwise_kernel_m_read_readvariableopJ
Fsavev2_adam_separable_conv2d_20_pointwise_kernel_m_read_readvariableop>
:savev2_adam_separable_conv2d_20_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_14_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_14_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_14_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_15_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_15_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_15_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_16_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_16_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_16_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_17_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_17_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_18_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_18_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_19_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_19_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_19_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_20_depthwise_kernel_v_read_readvariableopJ
Fsavev2_adam_separable_conv2d_20_pointwise_kernel_v_read_readvariableop>
:savev2_adam_separable_conv2d_20_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╤a
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:з*
dtype0*т`
value╪`B╒`зB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-17/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names█
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:з*
dtype0*ф
value┌B╫зB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЙP
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop?savev2_separable_conv2d_14_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_14_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop?savev2_separable_conv2d_15_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_15_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop?savev2_separable_conv2d_16_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_16_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop?savev2_separable_conv2d_17_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_17_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop?savev2_separable_conv2d_18_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_18_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop?savev2_separable_conv2d_19_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_19_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_19_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop?savev2_separable_conv2d_20_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_20_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_20_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableopFsavev2_adam_separable_conv2d_14_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_14_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_14_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableopFsavev2_adam_separable_conv2d_15_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_15_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_15_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableopFsavev2_adam_separable_conv2d_16_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_16_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_16_bias_m_read_readvariableop>savev2_adam_batch_normalization_19_gamma_m_read_readvariableop=savev2_adam_batch_normalization_19_beta_m_read_readvariableopFsavev2_adam_separable_conv2d_17_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_17_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_20_gamma_m_read_readvariableop=savev2_adam_batch_normalization_20_beta_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableopFsavev2_adam_separable_conv2d_18_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_18_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_21_gamma_m_read_readvariableop=savev2_adam_batch_normalization_21_beta_m_read_readvariableopFsavev2_adam_separable_conv2d_19_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_19_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_19_bias_m_read_readvariableop>savev2_adam_batch_normalization_22_gamma_m_read_readvariableop=savev2_adam_batch_normalization_22_beta_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableopFsavev2_adam_separable_conv2d_20_depthwise_kernel_m_read_readvariableopFsavev2_adam_separable_conv2d_20_pointwise_kernel_m_read_readvariableop:savev2_adam_separable_conv2d_20_bias_m_read_readvariableop>savev2_adam_batch_normalization_23_gamma_m_read_readvariableop=savev2_adam_batch_normalization_23_beta_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableopFsavev2_adam_separable_conv2d_14_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_14_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_14_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableopFsavev2_adam_separable_conv2d_15_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_15_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_15_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableopFsavev2_adam_separable_conv2d_16_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_16_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_16_bias_v_read_readvariableop>savev2_adam_batch_normalization_19_gamma_v_read_readvariableop=savev2_adam_batch_normalization_19_beta_v_read_readvariableopFsavev2_adam_separable_conv2d_17_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_17_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_20_gamma_v_read_readvariableop=savev2_adam_batch_normalization_20_beta_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableopFsavev2_adam_separable_conv2d_18_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_18_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_21_gamma_v_read_readvariableop=savev2_adam_batch_normalization_21_beta_v_read_readvariableopFsavev2_adam_separable_conv2d_19_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_19_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_19_bias_v_read_readvariableop>savev2_adam_batch_normalization_22_gamma_v_read_readvariableop=savev2_adam_batch_normalization_22_beta_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableopFsavev2_adam_separable_conv2d_20_depthwise_kernel_v_read_readvariableopFsavev2_adam_separable_conv2d_20_pointwise_kernel_v_read_readvariableop:savev2_adam_separable_conv2d_20_bias_v_read_readvariableop>savev2_adam_batch_normalization_23_gamma_v_read_readvariableop=savev2_adam_batch_normalization_23_beta_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *╕
dtypesн
к2з	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*е
_input_shapesУ
Р: :А:А:А:А:А:А:А:АА:А:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:АА:А:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:А╪:╪:╪:╪:╪:╪:╪:╪╪:╪:╪:╪:╪:╪:А╪:╪:╪:╪А:А:А:А:А:А:	А
:
: : : : : : : : : :А:А:А:А:А:АА:А:А:А:А:АА:А:А:А:АА:А:А:АА:А:А:А:А:АА:А:А:А:АА:А:А:А╪:╪:╪:╪:╪:╪╪:╪:╪:╪:А╪:╪:╪:╪А:А:А:А:	А
:
:А:А:А:А:А:АА:А:А:А:А:АА:А:А:А:АА:А:А:АА:А:А:А:А:АА:А:А:А:АА:А:А:А╪:╪:╪:╪:╪:╪╪:╪:╪:╪:А╪:╪:╪:╪А:А:А:А:	А
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А:.*
(
_output_shapes
:АА:!	

_output_shapes	
:А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:-)
'
_output_shapes
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А:.*
(
_output_shapes
:АА:! 

_output_shapes	
:А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:.%*
(
_output_shapes
:АА:!&

_output_shapes	
:А:-')
'
_output_shapes
:А:.(*
(
_output_shapes
:А╪:!)

_output_shapes	
:╪:!*

_output_shapes	
:╪:!+

_output_shapes	
:╪:!,

_output_shapes	
:╪:!-

_output_shapes	
:╪:-.)
'
_output_shapes
:╪:./*
(
_output_shapes
:╪╪:!0

_output_shapes	
:╪:!1

_output_shapes	
:╪:!2

_output_shapes	
:╪:!3

_output_shapes	
:╪:!4

_output_shapes	
:╪:.5*
(
_output_shapes
:А╪:!6

_output_shapes	
:╪:-7)
'
_output_shapes
:╪:.8*
(
_output_shapes
:╪А:!9

_output_shapes	
:А:!:

_output_shapes	
:А:!;

_output_shapes	
:А:!<

_output_shapes	
:А:!=

_output_shapes	
:А:%>!

_output_shapes
:	А
: ?

_output_shapes
:
:@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :-I)
'
_output_shapes
:А:!J

_output_shapes	
:А:!K

_output_shapes	
:А:!L

_output_shapes	
:А:-M)
'
_output_shapes
:А:.N*
(
_output_shapes
:АА:!O

_output_shapes	
:А:!P

_output_shapes	
:А:!Q

_output_shapes	
:А:-R)
'
_output_shapes
:А:.S*
(
_output_shapes
:АА:!T

_output_shapes	
:А:!U

_output_shapes	
:А:!V

_output_shapes	
:А:.W*
(
_output_shapes
:АА:!X

_output_shapes	
:А:-Y)
'
_output_shapes
:А:.Z*
(
_output_shapes
:АА:![

_output_shapes	
:А:!\

_output_shapes	
:А:!]

_output_shapes	
:А:-^)
'
_output_shapes
:А:._*
(
_output_shapes
:АА:!`

_output_shapes	
:А:!a

_output_shapes	
:А:!b

_output_shapes	
:А:.c*
(
_output_shapes
:АА:!d

_output_shapes	
:А:-e)
'
_output_shapes
:А:.f*
(
_output_shapes
:А╪:!g

_output_shapes	
:╪:!h

_output_shapes	
:╪:!i

_output_shapes	
:╪:-j)
'
_output_shapes
:╪:.k*
(
_output_shapes
:╪╪:!l

_output_shapes	
:╪:!m

_output_shapes	
:╪:!n

_output_shapes	
:╪:.o*
(
_output_shapes
:А╪:!p

_output_shapes	
:╪:-q)
'
_output_shapes
:╪:.r*
(
_output_shapes
:╪А:!s

_output_shapes	
:А:!t

_output_shapes	
:А:!u

_output_shapes	
:А:%v!

_output_shapes
:	А
: w

_output_shapes
:
:-x)
'
_output_shapes
:А:!y

_output_shapes	
:А:!z

_output_shapes	
:А:!{

_output_shapes	
:А:-|)
'
_output_shapes
:А:.}*
(
_output_shapes
:АА:!~

_output_shapes	
:А:!

_output_shapes	
:А:"А

_output_shapes	
:А:.Б)
'
_output_shapes
:А:/В*
(
_output_shapes
:АА:"Г

_output_shapes	
:А:"Д

_output_shapes	
:А:"Е

_output_shapes	
:А:/Ж*
(
_output_shapes
:АА:"З

_output_shapes	
:А:.И)
'
_output_shapes
:А:/Й*
(
_output_shapes
:АА:"К

_output_shapes	
:А:"Л

_output_shapes	
:А:"М

_output_shapes	
:А:.Н)
'
_output_shapes
:А:/О*
(
_output_shapes
:АА:"П

_output_shapes	
:А:"Р

_output_shapes	
:А:"С

_output_shapes	
:А:/Т*
(
_output_shapes
:АА:"У

_output_shapes	
:А:.Ф)
'
_output_shapes
:А:/Х*
(
_output_shapes
:А╪:"Ц

_output_shapes	
:╪:"Ч

_output_shapes	
:╪:"Ш

_output_shapes	
:╪:.Щ)
'
_output_shapes
:╪:/Ъ*
(
_output_shapes
:╪╪:"Ы

_output_shapes	
:╪:"Ь

_output_shapes	
:╪:"Э

_output_shapes	
:╪:/Ю*
(
_output_shapes
:А╪:"Я

_output_shapes	
:╪:.а)
'
_output_shapes
:╪:/б*
(
_output_shapes
:╪А:"в

_output_shapes	
:А:"г

_output_shapes	
:А:"д

_output_shapes	
:А:&е!

_output_shapes
:	А
:!ж

_output_shapes
:
:з

_output_shapes
: 
Ї
╓
7__inference_batch_normalization_18_layer_call_fn_435432

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4312852
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╧
в
*__inference_conv2d_11_layer_call_fn_436075

inputs#
unknown:А╪
	unknown_0:	╪
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_4325322
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╘
б
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_432570

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:         А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┌
L
0__inference_max_pooling2d_7_layer_call_fn_431677

inputs
identityя
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
GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_4316712
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь
б
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435517

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3э
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ў
╓
7__inference_batch_normalization_16_layer_call_fn_435141

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4309332
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_431451

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ї
╓
7__inference_batch_normalization_22_layer_call_fn_436030

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:	╪
	unknown_2:	╪
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4319252
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ╪2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ╪: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
р
╥
4__inference_separable_conv2d_20_layer_call_fn_432025

inputs"
unknown:╪%
	unknown_0:╪А
	unknown_1:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_4320132
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ╪: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ╪
 
_user_specified_nameinputs
╩
а
)__inference_conv2d_8_layer_call_fn_435056

inputs"
unknown:А
	unknown_0:	А
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_4321852
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╨
┼
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_431285

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1С
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ї
R
&__inference_add_6_layer_call_fn_435489
inputs_0
inputs_1
identity╪
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_4323302
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А:         А:Z V
0
_output_shapes
:         А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:         А
"
_user_specified_name
inputs/1
╜─
ю
C__inference_model_2_layer_call_and_return_conditional_losses_433539

inputs*
conv2d_8_433376:А
conv2d_8_433378:	А,
batch_normalization_16_433381:	А,
batch_normalization_16_433383:	А,
batch_normalization_16_433385:	А,
batch_normalization_16_433387:	А5
separable_conv2d_14_433392:А6
separable_conv2d_14_433394:АА)
separable_conv2d_14_433396:	А,
batch_normalization_17_433399:	А,
batch_normalization_17_433401:	А,
batch_normalization_17_433403:	А,
batch_normalization_17_433405:	А5
separable_conv2d_15_433409:А6
separable_conv2d_15_433411:АА)
separable_conv2d_15_433413:	А,
batch_normalization_18_433416:	А,
batch_normalization_18_433418:	А,
batch_normalization_18_433420:	А,
batch_normalization_18_433422:	А+
conv2d_9_433426:АА
conv2d_9_433428:	А5
separable_conv2d_16_433433:А6
separable_conv2d_16_433435:АА)
separable_conv2d_16_433437:	А,
batch_normalization_19_433440:	А,
batch_normalization_19_433442:	А,
batch_normalization_19_433444:	А,
batch_normalization_19_433446:	А5
separable_conv2d_17_433450:А6
separable_conv2d_17_433452:АА)
separable_conv2d_17_433454:	А,
batch_normalization_20_433457:	А,
batch_normalization_20_433459:	А,
batch_normalization_20_433461:	А,
batch_normalization_20_433463:	А,
conv2d_10_433467:АА
conv2d_10_433469:	А5
separable_conv2d_18_433474:А6
separable_conv2d_18_433476:А╪)
separable_conv2d_18_433478:	╪,
batch_normalization_21_433481:	╪,
batch_normalization_21_433483:	╪,
batch_normalization_21_433485:	╪,
batch_normalization_21_433487:	╪5
separable_conv2d_19_433491:╪6
separable_conv2d_19_433493:╪╪)
separable_conv2d_19_433495:	╪,
batch_normalization_22_433498:	╪,
batch_normalization_22_433500:	╪,
batch_normalization_22_433502:	╪,
batch_normalization_22_433504:	╪,
conv2d_11_433508:А╪
conv2d_11_433510:	╪5
separable_conv2d_20_433514:╪6
separable_conv2d_20_433516:╪А)
separable_conv2d_20_433518:	А,
batch_normalization_23_433521:	А,
batch_normalization_23_433523:	А,
batch_normalization_23_433525:	А,
batch_normalization_23_433527:	А!
dense_2_433533:	А

dense_2_433535:

identityИв.batch_normalization_16/StatefulPartitionedCallв.batch_normalization_17/StatefulPartitionedCallв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_10/StatefulPartitionedCallв!conv2d_11/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв+separable_conv2d_14/StatefulPartitionedCallв+separable_conv2d_15/StatefulPartitionedCallв+separable_conv2d_16/StatefulPartitionedCallв+separable_conv2d_17/StatefulPartitionedCallв+separable_conv2d_18/StatefulPartitionedCallв+separable_conv2d_19/StatefulPartitionedCallв+separable_conv2d_20/StatefulPartitionedCallm
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_2/Cast/xq
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_2/Cast_1/xИ
rescaling_2/mulMulinputsrescaling_2/Cast/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/mulЩ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*/
_output_shapes
:         @@2
rescaling_2/addн
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallrescaling_2/add:z:0conv2d_8_433376conv2d_8_433378*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_4321852"
 conv2d_8/StatefulPartitionedCall╔
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_16_433381batch_normalization_16_433383batch_normalization_16_433385batch_normalization_16_433387*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_43321520
.batch_normalization_16/StatefulPartitionedCallа
activation_16/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_4322232
activation_16/PartitionedCallП
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_4322302
activation_17/PartitionedCallХ
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0separable_conv2d_14_433392separable_conv2d_14_433394separable_conv2d_14_433396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_4310532-
+separable_conv2d_14/StatefulPartitionedCall╘
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0batch_normalization_17_433399batch_normalization_17_433401batch_normalization_17_433403batch_normalization_17_433405*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_43315920
.batch_normalization_17/StatefulPartitionedCallа
activation_18/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_4322712
activation_18/PartitionedCallХ
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0separable_conv2d_15_433409separable_conv2d_15_433411separable_conv2d_15_433413*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_4312072-
+separable_conv2d_15/StatefulPartitionedCall╘
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_433416batch_normalization_18_433418batch_normalization_18_433420batch_normalization_18_433422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_43310920
.batch_normalization_18/StatefulPartitionedCallж
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_4313512!
max_pooling2d_6/PartitionedCall└
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0conv2d_9_433426conv2d_9_433428*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_4323182"
 conv2d_9/StatefulPartitionedCallе
add_6/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_6_layer_call_and_return_conditional_losses_4323302
add_6/PartitionedCallЗ
activation_19/PartitionedCallPartitionedCalladd_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_4323372
activation_19/PartitionedCallХ
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0separable_conv2d_16_433433separable_conv2d_16_433435separable_conv2d_16_433437*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_4313732-
+separable_conv2d_16/StatefulPartitionedCall╘
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_433440batch_normalization_19_433442batch_normalization_19_433444batch_normalization_19_433446*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_43304220
.batch_normalization_19/StatefulPartitionedCallа
activation_20/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_4323782
activation_20/PartitionedCallХ
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0separable_conv2d_17_433450separable_conv2d_17_433452separable_conv2d_17_433454*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_4315272-
+separable_conv2d_17/StatefulPartitionedCall╘
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_433457batch_normalization_20_433459batch_normalization_20_433461batch_normalization_20_433463*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_43299220
.batch_normalization_20/StatefulPartitionedCallж
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_4316712!
max_pooling2d_7/PartitionedCall╜
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCalladd_6/PartitionedCall:output:0conv2d_10_433467conv2d_10_433469*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_4324252#
!conv2d_10/StatefulPartitionedCallж
add_7/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_7_layer_call_and_return_conditional_losses_4324372
add_7/PartitionedCallЗ
activation_21/PartitionedCallPartitionedCalladd_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_21_layer_call_and_return_conditional_losses_4324442
activation_21/PartitionedCallХ
+separable_conv2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0separable_conv2d_18_433474separable_conv2d_18_433476separable_conv2d_18_433478*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_4316932-
+separable_conv2d_18/StatefulPartitionedCall╘
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_433481batch_normalization_21_433483batch_normalization_21_433485batch_normalization_21_433487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_43292520
.batch_normalization_21/StatefulPartitionedCallа
activation_22/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_22_layer_call_and_return_conditional_losses_4324852
activation_22/PartitionedCallХ
+separable_conv2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0separable_conv2d_19_433491separable_conv2d_19_433493separable_conv2d_19_433495*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_4318472-
+separable_conv2d_19/StatefulPartitionedCall╘
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_433498batch_normalization_22_433500batch_normalization_22_433502batch_normalization_22_433504*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_43287520
.batch_normalization_22/StatefulPartitionedCallж
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_4319912!
max_pooling2d_8/PartitionedCall╜
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCalladd_7/PartitionedCall:output:0conv2d_11_433508conv2d_11_433510*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_4325322#
!conv2d_11/StatefulPartitionedCallж
add_8/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╪* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_8_layer_call_and_return_conditional_losses_4325442
add_8/PartitionedCallН
+separable_conv2d_20/StatefulPartitionedCallStatefulPartitionedCalladd_8/PartitionedCall:output:0separable_conv2d_20_433514separable_conv2d_20_433516separable_conv2d_20_433518*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_4320132-
+separable_conv2d_20/StatefulPartitionedCall╘
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall4separable_conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_433521batch_normalization_23_433523batch_normalization_23_433525batch_normalization_23_433527*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_43281420
.batch_normalization_23/StatefulPartitionedCallа
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_23_layer_call_and_return_conditional_losses_4325852
activation_23/PartitionedCallо
*global_average_pooling2d_2/PartitionedCallPartitionedCall&activation_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_4321582,
*global_average_pooling2d_2/PartitionedCallа
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4327722#
!dropout_2/StatefulPartitionedCall╢
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_433533dense_2_433535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4326062!
dense_2/StatefulPartitionedCallЪ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall,^separable_conv2d_18/StatefulPartitionedCall,^separable_conv2d_19/StatefulPartitionedCall,^separable_conv2d_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:         @@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall2Z
+separable_conv2d_18/StatefulPartitionedCall+separable_conv2d_18/StatefulPartitionedCall2Z
+separable_conv2d_19/StatefulPartitionedCall+separable_conv2d_19/StatefulPartitionedCall2Z
+separable_conv2d_20/StatefulPartitionedCall+separable_conv2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▓
serving_defaultЮ
C
input_38
serving_default_input_3:0         @@;
dense_20
StatefulPartitionedCall:0         
tensorflow/serving/predict:д╠

╓ы
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer_with_weights-12
layer-24
layer_with_weights-13
layer-25
layer-26
layer_with_weights-14
layer-27
layer_with_weights-15
layer-28
layer-29
layer_with_weights-16
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"layer_with_weights-18
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-19
&layer-37
'	optimizer
(regularization_losses
)trainable_variables
*	variables
+	keras_api
,
signatures
+м&call_and_return_all_conditional_losses
н__call__
о_default_save_signature"ыс
_tf_keras_network╬с{"name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Rescaling", "config": {"name": "rescaling_2", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "name": "rescaling_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["rescaling_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_16", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["activation_16", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_14", "inbound_nodes": [[["activation_17", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["separable_conv2d_14", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_15", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_18", "inbound_nodes": [[["separable_conv2d_15", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_6", "inbound_nodes": [[["batch_normalization_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["activation_16", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}], ["conv2d_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_16", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_16", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["separable_conv2d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_17", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_17", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["separable_conv2d_17", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_7", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["add_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["max_pooling2d_7", 0, 0, {}], ["conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_18", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_18", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["separable_conv2d_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_22", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_19", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 79}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_19", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["separable_conv2d_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_8", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["add_7", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "name": "add_8", "inbound_nodes": [[["max_pooling2d_8", 0, 0, {}], ["conv2d_11", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_20", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 94}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_20", "inbound_nodes": [[["add_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["separable_conv2d_20", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_2", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["global_average_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "shared_object_id": 108, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "float32", "input_3"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Rescaling", "config": {"name": "rescaling_2", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "name": "rescaling_2", "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["rescaling_2", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_16", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_14", "inbound_nodes": [[["activation_17", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 20}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["separable_conv2d_14", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_15", "inbound_nodes": [[["activation_18", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_18", "inbound_nodes": [[["separable_conv2d_15", 0, 0, {}]]], "shared_object_id": 32}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_6", "inbound_nodes": [[["batch_normalization_18", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "name": "add_6", "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}], ["conv2d_9", 0, 0, {}]]], "shared_object_id": 37}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["add_6", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_16", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_16", "inbound_nodes": [[["activation_19", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 45}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 47}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["separable_conv2d_16", 0, 0, {}]]], "shared_object_id": 48}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]], "shared_object_id": 49}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_17", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_17", "inbound_nodes": [[["activation_20", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 55}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 56}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 58}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["separable_conv2d_17", 0, 0, {}]]], "shared_object_id": 59}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_7", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]], "shared_object_id": 60}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["add_6", 0, 0, {}]]], "shared_object_id": 63}, {"class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "name": "add_7", "inbound_nodes": [[["max_pooling2d_7", 0, 0, {}], ["conv2d_10", 0, 0, {}]]], "shared_object_id": 64}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["add_7", 0, 0, {}]]], "shared_object_id": 65}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_18", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 69}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 66}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_18", "inbound_nodes": [[["activation_21", 0, 0, {}]]], "shared_object_id": 70}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 71}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 72}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 73}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 74}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["separable_conv2d_18", 0, 0, {}]]], "shared_object_id": 75}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_22", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]], "shared_object_id": 76}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_19", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 80}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 79}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 77}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 78}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_19", "inbound_nodes": [[["activation_22", 0, 0, {}]]], "shared_object_id": 81}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 82}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 83}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 84}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 85}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["separable_conv2d_19", 0, 0, {}]]], "shared_object_id": 86}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_8", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]], "shared_object_id": 87}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 88}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 89}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["add_7", 0, 0, {}]]], "shared_object_id": 90}, {"class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "name": "add_8", "inbound_nodes": [[["max_pooling2d_8", 0, 0, {}], ["conv2d_11", 0, 0, {}]]], "shared_object_id": 91}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_20", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 95}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 94}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 92}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 93}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_20", "inbound_nodes": [[["add_8", 0, 0, {}]]], "shared_object_id": 96}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 97}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 98}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 99}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 100}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["separable_conv2d_20", 0, 0, {}]]], "shared_object_id": 101}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]], "shared_object_id": 102}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_2", "inbound_nodes": [[["activation_23", 0, 0, {}]]], "shared_object_id": 103}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["global_average_pooling2d_2", 0, 0, {}]]], "shared_object_id": 104}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 105}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 106}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 107}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "SparseCategoricalCrossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 110}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.004999999888241291, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
∙"Ў
_tf_keras_input_layer╓{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
о
-	keras_api"Ь
_tf_keras_layerВ{"name": "rescaling_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "Rescaling", "config": {"name": "rescaling_2", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 1}
Г

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+п&call_and_return_all_conditional_losses
░__call__"▄	
_tf_keras_layer┬	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["rescaling_2", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 111}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
∙

4axis
	5gamma
6beta
7moving_mean
8moving_variance
9trainable_variables
:regularization_losses
;	variables
<	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"г	
_tf_keras_layerЙ	{"name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 112}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
м
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"Ы
_tf_keras_layerБ{"name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]], "shared_object_id": 10}
г
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+╡&call_and_return_all_conditional_losses
╢__call__"Т
_tf_keras_layer°{"name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 11}
┤
Edepthwise_kernel
Fpointwise_kernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
+╖&call_and_return_all_conditional_losses
╕__call__"э
_tf_keras_layer╙{"name": "separable_conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["activation_17", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}, "shared_object_id": 113}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Й
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"│	
_tf_keras_layerЩ	{"name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 20}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_14", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 114}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 256]}}
м
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"Ы
_tf_keras_layerБ{"name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]], "shared_object_id": 22}
┤
Ydepthwise_kernel
Zpointwise_kernel
[bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
+╜&call_and_return_all_conditional_losses
╛__call__"э
_tf_keras_layer╙{"name": "separable_conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["activation_18", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}, "shared_object_id": 115}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 256]}}
Й
`axis
	agamma
bbeta
cmoving_mean
dmoving_variance
etrainable_variables
fregularization_losses
g	variables
h	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"│	
_tf_keras_layerЩ	{"name": "batch_normalization_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_15", 0, 0, {}]]], "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 116}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 256]}}
ь
itrainable_variables
jregularization_losses
k	variables
l	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"█
_tf_keras_layer┴{"name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["batch_normalization_18", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 117}}
М

mkernel
nbias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
+├&call_and_return_all_conditional_losses
─__call__"х	
_tf_keras_layer╦	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 118}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
б
strainable_variables
tregularization_losses
u	variables
v	keras_api
+┼&call_and_return_all_conditional_losses
╞__call__"Р
_tf_keras_layerЎ{"name": "add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_6", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}], ["conv2d_9", 0, 0, {}]]], "shared_object_id": 37, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 16, 256]}, {"class_name": "TensorShape", "items": [null, 16, 16, 256]}]}
Ы
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+╟&call_and_return_all_conditional_losses
╚__call__"К
_tf_keras_layerЁ{"name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["add_6", 0, 0, {}]]], "shared_object_id": 38}
╢
{depthwise_kernel
|pointwise_kernel
}bias
~trainable_variables
regularization_losses
А	variables
Б	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"э
_tf_keras_layer╙{"name": "separable_conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_16", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["activation_19", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}, "shared_object_id": 119}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
Т
	Вaxis

Гgamma
	Дbeta
Еmoving_mean
Жmoving_variance
Зtrainable_variables
Иregularization_losses
Й	variables
К	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"│	
_tf_keras_layerЩ	{"name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 45}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 47}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_16", 0, 0, {}]]], "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 512}}, "shared_object_id": 120}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 512]}}
░
Лtrainable_variables
Мregularization_losses
Н	variables
О	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"Ы
_tf_keras_layerБ{"name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]], "shared_object_id": 49}
╗
Пdepthwise_kernel
Рpointwise_kernel
	Сbias
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
+╧&call_and_return_all_conditional_losses
╨__call__"э
_tf_keras_layer╙{"name": "separable_conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_17", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["activation_20", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}, "shared_object_id": 121}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 512]}}
Т
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ыtrainable_variables
Ьregularization_losses
Э	variables
Ю	keras_api
+╤&call_and_return_all_conditional_losses
╥__call__"│	
_tf_keras_layerЩ	{"name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 55}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 56}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 58}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_17", 0, 0, {}]]], "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 512}}, "shared_object_id": 122}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 512]}}
Ё
Яtrainable_variables
аregularization_losses
б	variables
в	keras_api
+╙&call_and_return_all_conditional_losses
╘__call__"█
_tf_keras_layer┴{"name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]], "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 123}}
М
гkernel
	дbias
еtrainable_variables
жregularization_losses
з	variables
и	keras_api
+╒&call_and_return_all_conditional_losses
╓__call__"▀	
_tf_keras_layer┼	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["add_6", 0, 0, {}]]], "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 124}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
в
йtrainable_variables
кregularization_losses
л	variables
м	keras_api
+╫&call_and_return_all_conditional_losses
╪__call__"Н
_tf_keras_layerє{"name": "add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_7", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["max_pooling2d_7", 0, 0, {}], ["conv2d_10", 0, 0, {}]]], "shared_object_id": 64, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 8, 8, 512]}, {"class_name": "TensorShape", "items": [null, 8, 8, 512]}]}
Я
нtrainable_variables
оregularization_losses
п	variables
░	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"К
_tf_keras_layerЁ{"name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["add_7", 0, 0, {}]]], "shared_object_id": 65}
╣
▒depthwise_kernel
▓pointwise_kernel
	│bias
┤trainable_variables
╡regularization_losses
╢	variables
╖	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"ы
_tf_keras_layer╤{"name": "separable_conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_18", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 69}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 66}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["activation_21", 0, 0, {}]]], "shared_object_id": 70, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}, "shared_object_id": 125}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 512]}}
Р
	╕axis

╣gamma
	║beta
╗moving_mean
╝moving_variance
╜trainable_variables
╛regularization_losses
┐	variables
└	keras_api
+▌&call_and_return_all_conditional_losses
▐__call__"▒	
_tf_keras_layerЧ	{"name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 71}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 72}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 73}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 74}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_18", 0, 0, {}]]], "shared_object_id": 75, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 728}}, "shared_object_id": 126}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 728]}}
░
┴trainable_variables
┬regularization_losses
├	variables
─	keras_api
+▀&call_and_return_all_conditional_losses
р__call__"Ы
_tf_keras_layerБ{"name": "activation_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]], "shared_object_id": 76}
╣
┼depthwise_kernel
╞pointwise_kernel
	╟bias
╚trainable_variables
╔regularization_losses
╩	variables
╦	keras_api
+с&call_and_return_all_conditional_losses
т__call__"ы
_tf_keras_layer╤{"name": "separable_conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_19", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 80}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 79}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 77}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 78}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["activation_22", 0, 0, {}]]], "shared_object_id": 81, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 728}}, "shared_object_id": 127}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 728]}}
Р
	╠axis

═gamma
	╬beta
╧moving_mean
╨moving_variance
╤trainable_variables
╥regularization_losses
╙	variables
╘	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"▒	
_tf_keras_layerЧ	{"name": "batch_normalization_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 82}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 83}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 84}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 85}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_19", 0, 0, {}]]], "shared_object_id": 86, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 728}}, "shared_object_id": 128}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 728]}}
Ё
╒trainable_variables
╓regularization_losses
╫	variables
╪	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"█
_tf_keras_layer┴{"name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]], "shared_object_id": 87, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 129}}
К
┘kernel
	┌bias
█trainable_variables
▄regularization_losses
▌	variables
▐	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"▌	
_tf_keras_layer├	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 728, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 88}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 89}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["add_7", 0, 0, {}]]], "shared_object_id": 90, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 130}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 512]}}
в
▀trainable_variables
рregularization_losses
с	variables
т	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"Н
_tf_keras_layerє{"name": "add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["max_pooling2d_8", 0, 0, {}], ["conv2d_11", 0, 0, {}]]], "shared_object_id": 91, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4, 4, 728]}, {"class_name": "TensorShape", "items": [null, 4, 4, 728]}]}
▓
уdepthwise_kernel
фpointwise_kernel
	хbias
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"ф
_tf_keras_layer╩{"name": "separable_conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_20", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 95}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 94}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 92}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 93}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "inbound_nodes": [[["add_8", 0, 0, {}]]], "shared_object_id": 96, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 728}}, "shared_object_id": 131}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 728]}}
Ф
	ъaxis

ыgamma
	ьbeta
эmoving_mean
юmoving_variance
яtrainable_variables
Ёregularization_losses
ё	variables
Є	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"╡	
_tf_keras_layerЫ	{"name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 97}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 98}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 99}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 100}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["separable_conv2d_20", 0, 0, {}]]], "shared_object_id": 101, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 1024}}, "shared_object_id": 132}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 1024]}}
▒
єtrainable_variables
Їregularization_losses
ї	variables
Ў	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"Ь
_tf_keras_layerВ{"name": "activation_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]], "shared_object_id": 102}
Б
ўtrainable_variables
°regularization_losses
∙	variables
·	keras_api
+ё&call_and_return_all_conditional_losses
Є__call__"ь
_tf_keras_layer╥{"name": "global_average_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["activation_23", 0, 0, {}]]], "shared_object_id": 103, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 133}}
├
√trainable_variables
№regularization_losses
¤	variables
■	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"о
_tf_keras_layerФ{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["global_average_pooling2d_2", 0, 0, {}]]], "shared_object_id": 104}
С	
 kernel
	Аbias
Бtrainable_variables
Вregularization_losses
Г	variables
Д	keras_api
+ї&call_and_return_all_conditional_losses
Ў__call__"ф
_tf_keras_layer╩{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 105}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 106}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 107, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}, "shared_object_id": 134}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
╝
	Еiter
Жbeta_1
Зbeta_2

Иdecay
Йlearning_rate.m╬/m╧5m╨6m╤Em╥Fm╙Gm╘Mm╒Nm╓Ym╫Zm╪[m┘am┌bm█mm▄nm▌{m▐|m▀}mр	Гmс	Дmт	Пmу	Рmф	Сmх	Чmц	Шmч	гmш	дmщ	▒mъ	▓mы	│mь	╣mэ	║mю	┼mя	╞mЁ	╟mё	═mЄ	╬mє	┘mЇ	┌mї	уmЎ	фmў	хm°	ыm∙	ьm·	 m√	Аm№.v¤/v■5v 6vАEvБFvВGvГMvДNvЕYvЖZvЗ[vИavЙbvКmvЛnvМ{vН|vО}vП	ГvР	ДvС	ПvТ	РvУ	СvФ	ЧvХ	ШvЦ	гvЧ	дvШ	▒vЩ	▓vЪ	│vЫ	╣vЬ	║vЭ	┼vЮ	╞vЯ	╟vа	═vб	╬vв	┘vг	┌vд	уvе	фvж	хvз	ыvи	ьvй	 vк	Аvл"
	optimizer
 "
trackable_list_wrapper
к
.0
/1
52
63
E4
F5
G6
M7
N8
Y9
Z10
[11
a12
b13
m14
n15
{16
|17
}18
Г19
Д20
П21
Р22
С23
Ч24
Ш25
г26
д27
▒28
▓29
│30
╣31
║32
┼33
╞34
╟35
═36
╬37
┘38
┌39
у40
ф41
х42
ы43
ь44
 45
А46"
trackable_list_wrapper
┤
.0
/1
52
63
74
85
E6
F7
G8
M9
N10
O11
P12
Y13
Z14
[15
a16
b17
c18
d19
m20
n21
{22
|23
}24
Г25
Д26
Е27
Ж28
П29
Р30
С31
Ч32
Ш33
Щ34
Ъ35
г36
д37
▒38
▓39
│40
╣41
║42
╗43
╝44
┼45
╞46
╟47
═48
╬49
╧50
╨51
┘52
┌53
у54
ф55
х56
ы57
ь58
э59
ю60
 61
А62"
trackable_list_wrapper
╙
 Кlayer_regularization_losses
(regularization_losses
Лnon_trainable_variables
)trainable_variables
Мlayer_metrics
*	variables
Нmetrics
Оlayers
н__call__
о_default_save_signature
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
-
ўserving_default"
signature_map
"
_generic_user_object
*:(А2conv2d_8/kernel
:А2conv2d_8/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
╡
 Пlayer_regularization_losses
Рnon_trainable_variables
0trainable_variables
1regularization_losses
Сlayer_metrics
2	variables
Тmetrics
Уlayers
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_16/gamma
*:(А2batch_normalization_16/beta
3:1А (2"batch_normalization_16/moving_mean
7:5А (2&batch_normalization_16/moving_variance
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
╡
 Фlayer_regularization_losses
Хnon_trainable_variables
9trainable_variables
:regularization_losses
Цlayer_metrics
;	variables
Чmetrics
Шlayers
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 Щlayer_regularization_losses
Ъnon_trainable_variables
=trainable_variables
>regularization_losses
Ыlayer_metrics
?	variables
Ьmetrics
Эlayers
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 Юlayer_regularization_losses
Яnon_trainable_variables
Atrainable_variables
Bregularization_losses
аlayer_metrics
C	variables
бmetrics
вlayers
╢__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_14/depthwise_kernel
@:>АА2$separable_conv2d_14/pointwise_kernel
':%А2separable_conv2d_14/bias
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
╡
 гlayer_regularization_losses
дnon_trainable_variables
Htrainable_variables
Iregularization_losses
еlayer_metrics
J	variables
жmetrics
зlayers
╕__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_17/gamma
*:(А2batch_normalization_17/beta
3:1А (2"batch_normalization_17/moving_mean
7:5А (2&batch_normalization_17/moving_variance
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
╡
 иlayer_regularization_losses
йnon_trainable_variables
Qtrainable_variables
Rregularization_losses
кlayer_metrics
S	variables
лmetrics
мlayers
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 нlayer_regularization_losses
оnon_trainable_variables
Utrainable_variables
Vregularization_losses
пlayer_metrics
W	variables
░metrics
▒layers
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_15/depthwise_kernel
@:>АА2$separable_conv2d_15/pointwise_kernel
':%А2separable_conv2d_15/bias
5
Y0
Z1
[2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
Y0
Z1
[2"
trackable_list_wrapper
╡
 ▓layer_regularization_losses
│non_trainable_variables
\trainable_variables
]regularization_losses
┤layer_metrics
^	variables
╡metrics
╢layers
╛__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_18/gamma
*:(А2batch_normalization_18/beta
3:1А (2"batch_normalization_18/moving_mean
7:5А (2&batch_normalization_18/moving_variance
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
a0
b1
c2
d3"
trackable_list_wrapper
╡
 ╖layer_regularization_losses
╕non_trainable_variables
etrainable_variables
fregularization_losses
╣layer_metrics
g	variables
║metrics
╗layers
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╝layer_regularization_losses
╜non_trainable_variables
itrainable_variables
jregularization_losses
╛layer_metrics
k	variables
┐metrics
└layers
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
+:)АА2conv2d_9/kernel
:А2conv2d_9/bias
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
╡
 ┴layer_regularization_losses
┬non_trainable_variables
otrainable_variables
pregularization_losses
├layer_metrics
q	variables
─metrics
┼layers
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╞layer_regularization_losses
╟non_trainable_variables
strainable_variables
tregularization_losses
╚layer_metrics
u	variables
╔metrics
╩layers
╞__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╦layer_regularization_losses
╠non_trainable_variables
wtrainable_variables
xregularization_losses
═layer_metrics
y	variables
╬metrics
╧layers
╚__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_16/depthwise_kernel
@:>АА2$separable_conv2d_16/pointwise_kernel
':%А2separable_conv2d_16/bias
5
{0
|1
}2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
{0
|1
}2"
trackable_list_wrapper
╢
 ╨layer_regularization_losses
╤non_trainable_variables
~trainable_variables
regularization_losses
╥layer_metrics
А	variables
╙metrics
╘layers
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_19/gamma
*:(А2batch_normalization_19/beta
3:1А (2"batch_normalization_19/moving_mean
7:5А (2&batch_normalization_19/moving_variance
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Г0
Д1
Е2
Ж3"
trackable_list_wrapper
╕
 ╒layer_regularization_losses
╓non_trainable_variables
Зtrainable_variables
Иregularization_losses
╫layer_metrics
Й	variables
╪metrics
┘layers
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 ┌layer_regularization_losses
█non_trainable_variables
Лtrainable_variables
Мregularization_losses
▄layer_metrics
Н	variables
▌metrics
▐layers
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_17/depthwise_kernel
@:>АА2$separable_conv2d_17/pointwise_kernel
':%А2separable_conv2d_17/bias
8
П0
Р1
С2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
П0
Р1
С2"
trackable_list_wrapper
╕
 ▀layer_regularization_losses
рnon_trainable_variables
Тtrainable_variables
Уregularization_losses
сlayer_metrics
Ф	variables
тmetrics
уlayers
╨__call__
+╧&call_and_return_all_conditional_losses
'╧"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_20/gamma
*:(А2batch_normalization_20/beta
3:1А (2"batch_normalization_20/moving_mean
7:5А (2&batch_normalization_20/moving_variance
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ч0
Ш1
Щ2
Ъ3"
trackable_list_wrapper
╕
 фlayer_regularization_losses
хnon_trainable_variables
Ыtrainable_variables
Ьregularization_losses
цlayer_metrics
Э	variables
чmetrics
шlayers
╥__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 щlayer_regularization_losses
ъnon_trainable_variables
Яtrainable_variables
аregularization_losses
ыlayer_metrics
б	variables
ьmetrics
эlayers
╘__call__
+╙&call_and_return_all_conditional_losses
'╙"call_and_return_conditional_losses"
_generic_user_object
,:*АА2conv2d_10/kernel
:А2conv2d_10/bias
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
╕
 юlayer_regularization_losses
яnon_trainable_variables
еtrainable_variables
жregularization_losses
Ёlayer_metrics
з	variables
ёmetrics
Єlayers
╓__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 єlayer_regularization_losses
Їnon_trainable_variables
йtrainable_variables
кregularization_losses
їlayer_metrics
л	variables
Ўmetrics
ўlayers
╪__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 °layer_regularization_losses
∙non_trainable_variables
нtrainable_variables
оregularization_losses
·layer_metrics
п	variables
√metrics
№layers
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
?:=А2$separable_conv2d_18/depthwise_kernel
@:>А╪2$separable_conv2d_18/pointwise_kernel
':%╪2separable_conv2d_18/bias
8
▒0
▓1
│2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
▒0
▓1
│2"
trackable_list_wrapper
╕
 ¤layer_regularization_losses
■non_trainable_variables
┤trainable_variables
╡regularization_losses
 layer_metrics
╢	variables
Аmetrics
Бlayers
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)╪2batch_normalization_21/gamma
*:(╪2batch_normalization_21/beta
3:1╪ (2"batch_normalization_21/moving_mean
7:5╪ (2&batch_normalization_21/moving_variance
0
╣0
║1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
╣0
║1
╗2
╝3"
trackable_list_wrapper
╕
 Вlayer_regularization_losses
Гnon_trainable_variables
╜trainable_variables
╛regularization_losses
Дlayer_metrics
┐	variables
Еmetrics
Жlayers
▐__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 Зlayer_regularization_losses
Иnon_trainable_variables
┴trainable_variables
┬regularization_losses
Йlayer_metrics
├	variables
Кmetrics
Лlayers
р__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
?:=╪2$separable_conv2d_19/depthwise_kernel
@:>╪╪2$separable_conv2d_19/pointwise_kernel
':%╪2separable_conv2d_19/bias
8
┼0
╞1
╟2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
┼0
╞1
╟2"
trackable_list_wrapper
╕
 Мlayer_regularization_losses
Нnon_trainable_variables
╚trainable_variables
╔regularization_losses
Оlayer_metrics
╩	variables
Пmetrics
Рlayers
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)╪2batch_normalization_22/gamma
*:(╪2batch_normalization_22/beta
3:1╪ (2"batch_normalization_22/moving_mean
7:5╪ (2&batch_normalization_22/moving_variance
0
═0
╬1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
═0
╬1
╧2
╨3"
trackable_list_wrapper
╕
 Сlayer_regularization_losses
Тnon_trainable_variables
╤trainable_variables
╥regularization_losses
Уlayer_metrics
╙	variables
Фmetrics
Хlayers
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 Цlayer_regularization_losses
Чnon_trainable_variables
╒trainable_variables
╓regularization_losses
Шlayer_metrics
╫	variables
Щmetrics
Ъlayers
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
,:*А╪2conv2d_11/kernel
:╪2conv2d_11/bias
0
┘0
┌1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
┘0
┌1"
trackable_list_wrapper
╕
 Ыlayer_regularization_losses
Ьnon_trainable_variables
█trainable_variables
▄regularization_losses
Эlayer_metrics
▌	variables
Юmetrics
Яlayers
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 аlayer_regularization_losses
бnon_trainable_variables
▀trainable_variables
рregularization_losses
вlayer_metrics
с	variables
гmetrics
дlayers
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
?:=╪2$separable_conv2d_20/depthwise_kernel
@:>╪А2$separable_conv2d_20/pointwise_kernel
':%А2separable_conv2d_20/bias
8
у0
ф1
х2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
у0
ф1
х2"
trackable_list_wrapper
╕
 еlayer_regularization_losses
жnon_trainable_variables
цtrainable_variables
чregularization_losses
зlayer_metrics
ш	variables
иmetrics
йlayers
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_23/gamma
*:(А2batch_normalization_23/beta
3:1А (2"batch_normalization_23/moving_mean
7:5А (2&batch_normalization_23/moving_variance
0
ы0
ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
ы0
ь1
э2
ю3"
trackable_list_wrapper
╕
 кlayer_regularization_losses
лnon_trainable_variables
яtrainable_variables
Ёregularization_losses
мlayer_metrics
ё	variables
нmetrics
оlayers
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 пlayer_regularization_losses
░non_trainable_variables
єtrainable_variables
Їregularization_losses
▒layer_metrics
ї	variables
▓metrics
│layers
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 ┤layer_regularization_losses
╡non_trainable_variables
ўtrainable_variables
°regularization_losses
╢layer_metrics
∙	variables
╖metrics
╕layers
Є__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 ╣layer_regularization_losses
║non_trainable_variables
√trainable_variables
№regularization_losses
╗layer_metrics
¤	variables
╝metrics
╜layers
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
!:	А
2dense_2/kernel
:
2dense_2/bias
0
 0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
 0
А1"
trackable_list_wrapper
╕
 ╛layer_regularization_losses
┐non_trainable_variables
Бtrainable_variables
Вregularization_losses
└layer_metrics
Г	variables
┴metrics
┬layers
Ў__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
а
70
81
O2
P3
c4
d5
Е6
Ж7
Щ8
Ъ9
╗10
╝11
╧12
╨13
э14
ю15"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
├0
─1"
trackable_list_wrapper
╞
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
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37"
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
.
70
81"
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
.
O0
P1"
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
.
c0
d1"
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
0
Е0
Ж1"
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
0
Щ0
Ъ1"
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
0
╗0
╝1"
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
0
╧0
╨1"
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
0
э0
ю1"
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
┘

┼total

╞count
╟	variables
╚	keras_api"Ю
_tf_keras_metricГ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 135}
д

╔total

╩count
╦
_fn_kwargs
╠	variables
═	keras_api"╪
_tf_keras_metric╜{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 110}
:  (2total
:  (2count
0
┼0
╞1"
trackable_list_wrapper
.
╟	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╔0
╩1"
trackable_list_wrapper
.
╠	variables"
_generic_user_object
/:-А2Adam/conv2d_8/kernel/m
!:А2Adam/conv2d_8/bias/m
0:.А2#Adam/batch_normalization_16/gamma/m
/:-А2"Adam/batch_normalization_16/beta/m
D:BА2+Adam/separable_conv2d_14/depthwise_kernel/m
E:CАА2+Adam/separable_conv2d_14/pointwise_kernel/m
,:*А2Adam/separable_conv2d_14/bias/m
0:.А2#Adam/batch_normalization_17/gamma/m
/:-А2"Adam/batch_normalization_17/beta/m
D:BА2+Adam/separable_conv2d_15/depthwise_kernel/m
E:CАА2+Adam/separable_conv2d_15/pointwise_kernel/m
,:*А2Adam/separable_conv2d_15/bias/m
0:.А2#Adam/batch_normalization_18/gamma/m
/:-А2"Adam/batch_normalization_18/beta/m
0:.АА2Adam/conv2d_9/kernel/m
!:А2Adam/conv2d_9/bias/m
D:BА2+Adam/separable_conv2d_16/depthwise_kernel/m
E:CАА2+Adam/separable_conv2d_16/pointwise_kernel/m
,:*А2Adam/separable_conv2d_16/bias/m
0:.А2#Adam/batch_normalization_19/gamma/m
/:-А2"Adam/batch_normalization_19/beta/m
D:BА2+Adam/separable_conv2d_17/depthwise_kernel/m
E:CАА2+Adam/separable_conv2d_17/pointwise_kernel/m
,:*А2Adam/separable_conv2d_17/bias/m
0:.А2#Adam/batch_normalization_20/gamma/m
/:-А2"Adam/batch_normalization_20/beta/m
1:/АА2Adam/conv2d_10/kernel/m
": А2Adam/conv2d_10/bias/m
D:BА2+Adam/separable_conv2d_18/depthwise_kernel/m
E:CА╪2+Adam/separable_conv2d_18/pointwise_kernel/m
,:*╪2Adam/separable_conv2d_18/bias/m
0:.╪2#Adam/batch_normalization_21/gamma/m
/:-╪2"Adam/batch_normalization_21/beta/m
D:B╪2+Adam/separable_conv2d_19/depthwise_kernel/m
E:C╪╪2+Adam/separable_conv2d_19/pointwise_kernel/m
,:*╪2Adam/separable_conv2d_19/bias/m
0:.╪2#Adam/batch_normalization_22/gamma/m
/:-╪2"Adam/batch_normalization_22/beta/m
1:/А╪2Adam/conv2d_11/kernel/m
": ╪2Adam/conv2d_11/bias/m
D:B╪2+Adam/separable_conv2d_20/depthwise_kernel/m
E:C╪А2+Adam/separable_conv2d_20/pointwise_kernel/m
,:*А2Adam/separable_conv2d_20/bias/m
0:.А2#Adam/batch_normalization_23/gamma/m
/:-А2"Adam/batch_normalization_23/beta/m
&:$	А
2Adam/dense_2/kernel/m
:
2Adam/dense_2/bias/m
/:-А2Adam/conv2d_8/kernel/v
!:А2Adam/conv2d_8/bias/v
0:.А2#Adam/batch_normalization_16/gamma/v
/:-А2"Adam/batch_normalization_16/beta/v
D:BА2+Adam/separable_conv2d_14/depthwise_kernel/v
E:CАА2+Adam/separable_conv2d_14/pointwise_kernel/v
,:*А2Adam/separable_conv2d_14/bias/v
0:.А2#Adam/batch_normalization_17/gamma/v
/:-А2"Adam/batch_normalization_17/beta/v
D:BА2+Adam/separable_conv2d_15/depthwise_kernel/v
E:CАА2+Adam/separable_conv2d_15/pointwise_kernel/v
,:*А2Adam/separable_conv2d_15/bias/v
0:.А2#Adam/batch_normalization_18/gamma/v
/:-А2"Adam/batch_normalization_18/beta/v
0:.АА2Adam/conv2d_9/kernel/v
!:А2Adam/conv2d_9/bias/v
D:BА2+Adam/separable_conv2d_16/depthwise_kernel/v
E:CАА2+Adam/separable_conv2d_16/pointwise_kernel/v
,:*А2Adam/separable_conv2d_16/bias/v
0:.А2#Adam/batch_normalization_19/gamma/v
/:-А2"Adam/batch_normalization_19/beta/v
D:BА2+Adam/separable_conv2d_17/depthwise_kernel/v
E:CАА2+Adam/separable_conv2d_17/pointwise_kernel/v
,:*А2Adam/separable_conv2d_17/bias/v
0:.А2#Adam/batch_normalization_20/gamma/v
/:-А2"Adam/batch_normalization_20/beta/v
1:/АА2Adam/conv2d_10/kernel/v
": А2Adam/conv2d_10/bias/v
D:BА2+Adam/separable_conv2d_18/depthwise_kernel/v
E:CА╪2+Adam/separable_conv2d_18/pointwise_kernel/v
,:*╪2Adam/separable_conv2d_18/bias/v
0:.╪2#Adam/batch_normalization_21/gamma/v
/:-╪2"Adam/batch_normalization_21/beta/v
D:B╪2+Adam/separable_conv2d_19/depthwise_kernel/v
E:C╪╪2+Adam/separable_conv2d_19/pointwise_kernel/v
,:*╪2Adam/separable_conv2d_19/bias/v
0:.╪2#Adam/batch_normalization_22/gamma/v
/:-╪2"Adam/batch_normalization_22/beta/v
1:/А╪2Adam/conv2d_11/kernel/v
": ╪2Adam/conv2d_11/bias/v
D:B╪2+Adam/separable_conv2d_20/depthwise_kernel/v
E:C╪А2+Adam/separable_conv2d_20/pointwise_kernel/v
,:*А2Adam/separable_conv2d_20/bias/v
0:.А2#Adam/batch_normalization_23/gamma/v
/:-А2"Adam/batch_normalization_23/beta/v
&:$	А
2Adam/dense_2/kernel/v
:
2Adam/dense_2/bias/v
┌2╫
C__inference_model_2_layer_call_and_return_conditional_losses_434523
C__inference_model_2_layer_call_and_return_conditional_losses_434775
C__inference_model_2_layer_call_and_return_conditional_losses_433969
C__inference_model_2_layer_call_and_return_conditional_losses_434139└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
(__inference_model_2_layer_call_fn_432742
(__inference_model_2_layer_call_fn_434906
(__inference_model_2_layer_call_fn_435037
(__inference_model_2_layer_call_fn_433799└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ч2ф
!__inference__wrapped_model_430911╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_3         @@
ю2ы
D__inference_conv2d_8_layer_call_and_return_conditional_losses_435047в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_8_layer_call_fn_435056в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
К2З
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435074
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435092
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435110
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435128┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_16_layer_call_fn_435141
7__inference_batch_normalization_16_layer_call_fn_435154
7__inference_batch_normalization_16_layer_call_fn_435167
7__inference_batch_normalization_16_layer_call_fn_435180┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_activation_16_layer_call_and_return_conditional_losses_435185в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_16_layer_call_fn_435190в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_activation_17_layer_call_and_return_conditional_losses_435195в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_17_layer_call_fn_435200в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_431053╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ф2С
4__inference_separable_conv2d_14_layer_call_fn_431065╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
К2З
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435218
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435236
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435254
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435272┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_17_layer_call_fn_435285
7__inference_batch_normalization_17_layer_call_fn_435298
7__inference_batch_normalization_17_layer_call_fn_435311
7__inference_batch_normalization_17_layer_call_fn_435324┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_activation_18_layer_call_and_return_conditional_losses_435329в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_18_layer_call_fn_435334в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_431207╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ф2С
4__inference_separable_conv2d_15_layer_call_fn_431219╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
К2З
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435352
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435370
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435388
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435406┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_18_layer_call_fn_435419
7__inference_batch_normalization_18_layer_call_fn_435432
7__inference_batch_normalization_18_layer_call_fn_435445
7__inference_batch_normalization_18_layer_call_fn_435458┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
│2░
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_431351р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_6_layer_call_fn_431357р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
ю2ы
D__inference_conv2d_9_layer_call_and_return_conditional_losses_435468в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_9_layer_call_fn_435477в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_add_6_layer_call_and_return_conditional_losses_435483в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_add_6_layer_call_fn_435489в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_activation_19_layer_call_and_return_conditional_losses_435494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_19_layer_call_fn_435499в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_431373╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ф2С
4__inference_separable_conv2d_16_layer_call_fn_431385╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
К2З
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435517
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435535
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435553
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435571┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_19_layer_call_fn_435584
7__inference_batch_normalization_19_layer_call_fn_435597
7__inference_batch_normalization_19_layer_call_fn_435610
7__inference_batch_normalization_19_layer_call_fn_435623┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_activation_20_layer_call_and_return_conditional_losses_435628в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_20_layer_call_fn_435633в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_431527╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ф2С
4__inference_separable_conv2d_17_layer_call_fn_431539╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
К2З
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435651
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435669
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435687
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435705┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_20_layer_call_fn_435718
7__inference_batch_normalization_20_layer_call_fn_435731
7__inference_batch_normalization_20_layer_call_fn_435744
7__inference_batch_normalization_20_layer_call_fn_435757┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
│2░
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_431671р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_7_layer_call_fn_431677р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_10_layer_call_and_return_conditional_losses_435767в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_10_layer_call_fn_435776в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_add_7_layer_call_and_return_conditional_losses_435782в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_add_7_layer_call_fn_435788в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_activation_21_layer_call_and_return_conditional_losses_435793в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_21_layer_call_fn_435798в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_431693╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ф2С
4__inference_separable_conv2d_18_layer_call_fn_431705╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
К2З
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435816
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435834
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435852
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435870┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_21_layer_call_fn_435883
7__inference_batch_normalization_21_layer_call_fn_435896
7__inference_batch_normalization_21_layer_call_fn_435909
7__inference_batch_normalization_21_layer_call_fn_435922┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_activation_22_layer_call_and_return_conditional_losses_435927в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_22_layer_call_fn_435932в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_431847╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           ╪
Ф2С
4__inference_separable_conv2d_19_layer_call_fn_431859╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           ╪
К2З
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435950
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435968
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435986
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_436004┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_22_layer_call_fn_436017
7__inference_batch_normalization_22_layer_call_fn_436030
7__inference_batch_normalization_22_layer_call_fn_436043
7__inference_batch_normalization_22_layer_call_fn_436056┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
│2░
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_431991р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_8_layer_call_fn_431997р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_11_layer_call_and_return_conditional_losses_436066в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_11_layer_call_fn_436075в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_add_8_layer_call_and_return_conditional_losses_436081в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_add_8_layer_call_fn_436087в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2м
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_432013╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           ╪
Ф2С
4__inference_separable_conv2d_20_layer_call_fn_432025╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           ╪
К2З
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436105
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436123
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436141
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436159┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_23_layer_call_fn_436172
7__inference_batch_normalization_23_layer_call_fn_436185
7__inference_batch_normalization_23_layer_call_fn_436198
7__inference_batch_normalization_23_layer_call_fn_436211┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_activation_23_layer_call_and_return_conditional_losses_436216в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_activation_23_layer_call_fn_436221в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╛2╗
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_432158р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
;__inference_global_average_pooling2d_2_layer_call_fn_432164р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╚2┼
E__inference_dropout_2_layer_call_and_return_conditional_losses_436226
E__inference_dropout_2_layer_call_and_return_conditional_losses_436238┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_2_layer_call_fn_436243
*__inference_dropout_2_layer_call_fn_436248┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_436259в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_2_layer_call_fn_436268в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦B╚
$__inference_signature_wrapper_434278input_3"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ·
!__inference__wrapped_model_430911╘e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А8в5
.в+
)К&
input_3         @@
к "1к.
,
dense_2!К
dense_2         
╖
I__inference_activation_16_layer_call_and_return_conditional_losses_435185j8в5
.в+
)К&
inputs           А
к ".в+
$К!
0           А
Ъ П
.__inference_activation_16_layer_call_fn_435190]8в5
.в+
)К&
inputs           А
к "!К           А╖
I__inference_activation_17_layer_call_and_return_conditional_losses_435195j8в5
.в+
)К&
inputs           А
к ".в+
$К!
0           А
Ъ П
.__inference_activation_17_layer_call_fn_435200]8в5
.в+
)К&
inputs           А
к "!К           А╖
I__inference_activation_18_layer_call_and_return_conditional_losses_435329j8в5
.в+
)К&
inputs           А
к ".в+
$К!
0           А
Ъ П
.__inference_activation_18_layer_call_fn_435334]8в5
.в+
)К&
inputs           А
к "!К           А╖
I__inference_activation_19_layer_call_and_return_conditional_losses_435494j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
.__inference_activation_19_layer_call_fn_435499]8в5
.в+
)К&
inputs         А
к "!К         А╖
I__inference_activation_20_layer_call_and_return_conditional_losses_435628j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
.__inference_activation_20_layer_call_fn_435633]8в5
.в+
)К&
inputs         А
к "!К         А╖
I__inference_activation_21_layer_call_and_return_conditional_losses_435793j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
.__inference_activation_21_layer_call_fn_435798]8в5
.в+
)К&
inputs         А
к "!К         А╖
I__inference_activation_22_layer_call_and_return_conditional_losses_435927j8в5
.в+
)К&
inputs         ╪
к ".в+
$К!
0         ╪
Ъ П
.__inference_activation_22_layer_call_fn_435932]8в5
.в+
)К&
inputs         ╪
к "!К         ╪╖
I__inference_activation_23_layer_call_and_return_conditional_losses_436216j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
.__inference_activation_23_layer_call_fn_436221]8в5
.в+
)К&
inputs         А
к "!К         Аф
A__inference_add_6_layer_call_and_return_conditional_losses_435483Юlвi
bв_
]ЪZ
+К(
inputs/0         А
+К(
inputs/1         А
к ".в+
$К!
0         А
Ъ ╝
&__inference_add_6_layer_call_fn_435489Сlвi
bв_
]ЪZ
+К(
inputs/0         А
+К(
inputs/1         А
к "!К         Аф
A__inference_add_7_layer_call_and_return_conditional_losses_435782Юlвi
bв_
]ЪZ
+К(
inputs/0         А
+К(
inputs/1         А
к ".в+
$К!
0         А
Ъ ╝
&__inference_add_7_layer_call_fn_435788Сlвi
bв_
]ЪZ
+К(
inputs/0         А
+К(
inputs/1         А
к "!К         Аф
A__inference_add_8_layer_call_and_return_conditional_losses_436081Юlвi
bв_
]ЪZ
+К(
inputs/0         ╪
+К(
inputs/1         ╪
к ".в+
$К!
0         ╪
Ъ ╝
&__inference_add_8_layer_call_fn_436087Сlвi
bв_
]ЪZ
+К(
inputs/0         ╪
+К(
inputs/1         ╪
к "!К         ╪я
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435074Ш5678NвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435092Ш5678NвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╩
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435110t5678<в9
2в/
)К&
inputs           А
p 
к ".в+
$К!
0           А
Ъ ╩
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_435128t5678<в9
2в/
)К&
inputs           А
p
к ".в+
$К!
0           А
Ъ ╟
7__inference_batch_normalization_16_layer_call_fn_435141Л5678NвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╟
7__inference_batch_normalization_16_layer_call_fn_435154Л5678NвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Ав
7__inference_batch_normalization_16_layer_call_fn_435167g5678<в9
2в/
)К&
inputs           А
p 
к "!К           Ав
7__inference_batch_normalization_16_layer_call_fn_435180g5678<в9
2в/
)К&
inputs           А
p
к "!К           Ая
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435218ШMNOPNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435236ШMNOPNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╩
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435254tMNOP<в9
2в/
)К&
inputs           А
p 
к ".в+
$К!
0           А
Ъ ╩
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_435272tMNOP<в9
2в/
)К&
inputs           А
p
к ".в+
$К!
0           А
Ъ ╟
7__inference_batch_normalization_17_layer_call_fn_435285ЛMNOPNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╟
7__inference_batch_normalization_17_layer_call_fn_435298ЛMNOPNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Ав
7__inference_batch_normalization_17_layer_call_fn_435311gMNOP<в9
2в/
)К&
inputs           А
p 
к "!К           Ав
7__inference_batch_normalization_17_layer_call_fn_435324gMNOP<в9
2в/
)К&
inputs           А
p
к "!К           Ая
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435352ШabcdNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435370ШabcdNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╩
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435388tabcd<в9
2в/
)К&
inputs           А
p 
к ".в+
$К!
0           А
Ъ ╩
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_435406tabcd<в9
2в/
)К&
inputs           А
p
к ".в+
$К!
0           А
Ъ ╟
7__inference_batch_normalization_18_layer_call_fn_435419ЛabcdNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╟
7__inference_batch_normalization_18_layer_call_fn_435432ЛabcdNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Ав
7__inference_batch_normalization_18_layer_call_fn_435445gabcd<в9
2в/
)К&
inputs           А
p 
к "!К           Ав
7__inference_batch_normalization_18_layer_call_fn_435458gabcd<в9
2в/
)К&
inputs           А
p
к "!К           Ає
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435517ЬГДЕЖNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ є
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435535ЬГДЕЖNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╬
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435553xГДЕЖ<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ╬
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_435571xГДЕЖ<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ ╦
7__inference_batch_normalization_19_layer_call_fn_435584ПГДЕЖNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╦
7__inference_batch_normalization_19_layer_call_fn_435597ПГДЕЖNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Аж
7__inference_batch_normalization_19_layer_call_fn_435610kГДЕЖ<в9
2в/
)К&
inputs         А
p 
к "!К         Аж
7__inference_batch_normalization_19_layer_call_fn_435623kГДЕЖ<в9
2в/
)К&
inputs         А
p
к "!К         Ає
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435651ЬЧШЩЪNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ є
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435669ЬЧШЩЪNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╬
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435687xЧШЩЪ<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ╬
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_435705xЧШЩЪ<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ ╦
7__inference_batch_normalization_20_layer_call_fn_435718ПЧШЩЪNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╦
7__inference_batch_normalization_20_layer_call_fn_435731ПЧШЩЪNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Аж
7__inference_batch_normalization_20_layer_call_fn_435744kЧШЩЪ<в9
2в/
)К&
inputs         А
p 
к "!К         Аж
7__inference_batch_normalization_20_layer_call_fn_435757kЧШЩЪ<в9
2в/
)К&
inputs         А
p
к "!К         Ає
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435816Ь╣║╗╝NвK
DвA
;К8
inputs,                           ╪
p 
к "@в=
6К3
0,                           ╪
Ъ є
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435834Ь╣║╗╝NвK
DвA
;К8
inputs,                           ╪
p
к "@в=
6К3
0,                           ╪
Ъ ╬
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435852x╣║╗╝<в9
2в/
)К&
inputs         ╪
p 
к ".в+
$К!
0         ╪
Ъ ╬
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_435870x╣║╗╝<в9
2в/
)К&
inputs         ╪
p
к ".в+
$К!
0         ╪
Ъ ╦
7__inference_batch_normalization_21_layer_call_fn_435883П╣║╗╝NвK
DвA
;К8
inputs,                           ╪
p 
к "3К0,                           ╪╦
7__inference_batch_normalization_21_layer_call_fn_435896П╣║╗╝NвK
DвA
;К8
inputs,                           ╪
p
к "3К0,                           ╪ж
7__inference_batch_normalization_21_layer_call_fn_435909k╣║╗╝<в9
2в/
)К&
inputs         ╪
p 
к "!К         ╪ж
7__inference_batch_normalization_21_layer_call_fn_435922k╣║╗╝<в9
2в/
)К&
inputs         ╪
p
к "!К         ╪є
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435950Ь═╬╧╨NвK
DвA
;К8
inputs,                           ╪
p 
к "@в=
6К3
0,                           ╪
Ъ є
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435968Ь═╬╧╨NвK
DвA
;К8
inputs,                           ╪
p
к "@в=
6К3
0,                           ╪
Ъ ╬
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_435986x═╬╧╨<в9
2в/
)К&
inputs         ╪
p 
к ".в+
$К!
0         ╪
Ъ ╬
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_436004x═╬╧╨<в9
2в/
)К&
inputs         ╪
p
к ".в+
$К!
0         ╪
Ъ ╦
7__inference_batch_normalization_22_layer_call_fn_436017П═╬╧╨NвK
DвA
;К8
inputs,                           ╪
p 
к "3К0,                           ╪╦
7__inference_batch_normalization_22_layer_call_fn_436030П═╬╧╨NвK
DвA
;К8
inputs,                           ╪
p
к "3К0,                           ╪ж
7__inference_batch_normalization_22_layer_call_fn_436043k═╬╧╨<в9
2в/
)К&
inputs         ╪
p 
к "!К         ╪ж
7__inference_batch_normalization_22_layer_call_fn_436056k═╬╧╨<в9
2в/
)К&
inputs         ╪
p
к "!К         ╪є
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436105ЬыьэюNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ є
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436123ЬыьэюNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╬
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436141xыьэю<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ╬
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_436159xыьэю<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ ╦
7__inference_batch_normalization_23_layer_call_fn_436172ПыьэюNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╦
7__inference_batch_normalization_23_layer_call_fn_436185ПыьэюNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Аж
7__inference_batch_normalization_23_layer_call_fn_436198kыьэю<в9
2в/
)К&
inputs         А
p 
к "!К         Аж
7__inference_batch_normalization_23_layer_call_fn_436211kыьэю<в9
2в/
)К&
inputs         А
p
к "!К         А╣
E__inference_conv2d_10_layer_call_and_return_conditional_losses_435767pгд8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ С
*__inference_conv2d_10_layer_call_fn_435776cгд8в5
.в+
)К&
inputs         А
к "!К         А╣
E__inference_conv2d_11_layer_call_and_return_conditional_losses_436066p┘┌8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         ╪
Ъ С
*__inference_conv2d_11_layer_call_fn_436075c┘┌8в5
.в+
)К&
inputs         А
к "!К         ╪╡
D__inference_conv2d_8_layer_call_and_return_conditional_losses_435047m./7в4
-в*
(К%
inputs         @@
к ".в+
$К!
0           А
Ъ Н
)__inference_conv2d_8_layer_call_fn_435056`./7в4
-в*
(К%
inputs         @@
к "!К           А╢
D__inference_conv2d_9_layer_call_and_return_conditional_losses_435468nmn8в5
.в+
)К&
inputs           А
к ".в+
$К!
0         А
Ъ О
)__inference_conv2d_9_layer_call_fn_435477amn8в5
.в+
)К&
inputs           А
к "!К         Аж
C__inference_dense_2_layer_call_and_return_conditional_losses_436259_ А0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ ~
(__inference_dense_2_layer_call_fn_436268R А0в-
&в#
!К
inputs         А
к "К         
з
E__inference_dropout_2_layer_call_and_return_conditional_losses_436226^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ з
E__inference_dropout_2_layer_call_and_return_conditional_losses_436238^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ 
*__inference_dropout_2_layer_call_fn_436243Q4в1
*в'
!К
inputs         А
p 
к "К         А
*__inference_dropout_2_layer_call_fn_436248Q4в1
*в'
!К
inputs         А
p
к "К         А▀
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_432158ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ╢
;__inference_global_average_pooling2d_2_layer_call_fn_432164wRвO
HвE
CК@
inputs4                                    
к "!К                  ю
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_431351ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_6_layer_call_fn_431357СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_431671ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_7_layer_call_fn_431677СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_431991ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_8_layer_call_fn_431997СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ш
C__inference_model_2_layer_call_and_return_conditional_losses_433969╨e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А@в=
6в3
)К&
input_3         @@
p 

 
к "%в"
К
0         

Ъ Ш
C__inference_model_2_layer_call_and_return_conditional_losses_434139╨e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А@в=
6в3
)К&
input_3         @@
p

 
к "%в"
К
0         

Ъ Ч
C__inference_model_2_layer_call_and_return_conditional_losses_434523╧e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А?в<
5в2
(К%
inputs         @@
p 

 
к "%в"
К
0         

Ъ Ч
C__inference_model_2_layer_call_and_return_conditional_losses_434775╧e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А?в<
5в2
(К%
inputs         @@
p

 
к "%в"
К
0         

Ъ Ё
(__inference_model_2_layer_call_fn_432742├e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А@в=
6в3
)К&
input_3         @@
p 

 
к "К         
Ё
(__inference_model_2_layer_call_fn_433799├e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А@в=
6в3
)К&
input_3         @@
p

 
к "К         
я
(__inference_model_2_layer_call_fn_434906┬e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А?в<
5в2
(К%
inputs         @@
p 

 
к "К         
я
(__inference_model_2_layer_call_fn_435037┬e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю А?в<
5в2
(К%
inputs         @@
p

 
к "К         
ч
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_431053УEFGJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┐
4__inference_separable_conv2d_14_layer_call_fn_431065ЖEFGJвG
@в=
;К8
inputs,                           А
к "3К0,                           Ач
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_431207УYZ[JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┐
4__inference_separable_conv2d_15_layer_call_fn_431219ЖYZ[JвG
@в=
;К8
inputs,                           А
к "3К0,                           Ач
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_431373У{|}JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┐
4__inference_separable_conv2d_16_layer_call_fn_431385Ж{|}JвG
@в=
;К8
inputs,                           А
к "3К0,                           Аъ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_431527ЦПРСJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┬
4__inference_separable_conv2d_17_layer_call_fn_431539ЙПРСJвG
@в=
;К8
inputs,                           А
к "3К0,                           Аъ
O__inference_separable_conv2d_18_layer_call_and_return_conditional_losses_431693Ц▒▓│JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           ╪
Ъ ┬
4__inference_separable_conv2d_18_layer_call_fn_431705Й▒▓│JвG
@в=
;К8
inputs,                           А
к "3К0,                           ╪ъ
O__inference_separable_conv2d_19_layer_call_and_return_conditional_losses_431847Ц┼╞╟JвG
@в=
;К8
inputs,                           ╪
к "@в=
6К3
0,                           ╪
Ъ ┬
4__inference_separable_conv2d_19_layer_call_fn_431859Й┼╞╟JвG
@в=
;К8
inputs,                           ╪
к "3К0,                           ╪ъ
O__inference_separable_conv2d_20_layer_call_and_return_conditional_losses_432013ЦуфхJвG
@в=
;К8
inputs,                           ╪
к "@в=
6К3
0,                           А
Ъ ┬
4__inference_separable_conv2d_20_layer_call_fn_432025ЙуфхJвG
@в=
;К8
inputs,                           ╪
к "3К0,                           АИ
$__inference_signature_wrapper_434278▀e./5678EFGMNOPYZ[abcdmn{|}ГДЕЖПРСЧШЩЪгд▒▓│╣║╗╝┼╞╟═╬╧╨┘┌уфхыьэю АCв@
в 
9к6
4
input_3)К&
input_3         @@"1к.
,
dense_2!К
dense_2         
