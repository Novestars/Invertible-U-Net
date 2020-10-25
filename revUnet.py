import warnings

from typing import Union, Iterable, Callable, Any, Tuple, Sized, List, Optional
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import set_grad_enabled
#from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Function

try:
    pytorch_version_one_and_above = int(torch.__version__[0]) > 0
except TypeError:
    pytorch_version_one_and_above = True

def create_standard_module(in_channels, **kwargs):
    dim = kwargs.pop('dim', 2)
    block_depth = kwargs.pop('block_depth', 1)
    num_channels = get_num_channels(in_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels

    module_index = kwargs.pop('module_index', 0)
    # For odd number of channels, this switches the roles of input and output
    # channels at every other layer, e.g. 1->2, then 2->1.
    if np.mod(module_index, 2) == 0:
        (num_F_in_channels, num_F_out_channels) = (
            num_F_out_channels, num_F_in_channels
        )
    return StandardAdditiveCoupling(
        F=StandardBlock(
            dim,
            num_F_in_channels,
            num_F_out_channels,
            block_depth=block_depth),
        channel_split_pos=num_F_out_channels
    )


class iUNet(nn.Module):
    """Fully-invertible U-Net (iUNet).
    This model can be used for memory-efficient backpropagation, e.g. in
    high-dimensional (such as 3D) segmentation tasks.
    :param in_channels:
        The number of input channels, which is then also the number of output
        channels. Can also be the complete input shape (without batch
        dimension).
    :param architecture:
        Determines the number of invertible layers at each
        resolution (both left and right), e.g. ``[2,3,4]`` results in the
        following structure::
            2-----2
             3---3
              4-4
    :param dim: Either ``1``, ``2`` or ``3``, signifying whether a 1D, 2D or 3D
        invertible U-Net should be created.
    :param create_module_fn:
        Function which outputs an invertible layer. This layer
        should be a ``torch.nn.Module`` with a method ``forward(*x)``
        and a method ``inverse(*x)``. ``create_module_fn`` should have the
        signature ``create_module_fn(in_channels, **kwargs)``.
        Additional keyword arguments passed on via ``kwargs`` are
        ``dim`` (whether this is a 1D, 2D or 3D iUNet), the coordinates
        of the specific module within the iUNet (``LR``, ``level`` and
        ``module_index``) as well as ``architecture``. By default, this creates
        an additive coupling layer, whose block consists of a number of
        convolutional layers, followed by a `leaky ReLU` activation function
        and an instance normalization layer. The number of blocks can be
        controlled by setting ``"block_depth"`` in ``module_kwargs``.
    :param module_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to ``create_module_fn``.
    :param slice_mode:
        Controls the fraction of channels, which gets invertibly
        downsampled. Together with invertible downsampling
        Currently supported modes: ``"double"``, ``"constant"``.
        Defaults to ``"double"``.
    :param learnable_resampling:
        Whether to train the invertible learnable up- and downsampling
        or to leave it at the initialized values.
        Defaults to ``True``.
    :param resampling_stride:
        Controls the stride of the invertible up- and downsampling.
        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        For example: ``2`` would result in a up-/downsampling with a factor of 2
        along each dimension; ``(2,1,4)`` would apply (at every
        resampling) a factor of 2, 1 and 4 for the height, width and depth
        dimensions respectively, whereas for a 3D iUNet with 3 up-/downsampling
        stages, ``[(2,1,3), (2,2,2), (4,3,1)]`` would result in different
        strides at different up-/downsampling stages.
    :param resampling_method:
        Chooses the method for parametrizing orthogonal matrices for
        invertible up- and downsampling. Can be either ``"exp"`` (i.e.
        exponentiation of skew-symmetric matrices) or ``"cayley"`` (i.e.
        the Cayley transform, acting on skew-symmetric matrices).
        Defaults to ``"cayley"``.
    :param resampling_init:
        Sets the initialization for the learnable up- and downsampling
        operators. Can be ``"haar"``, ``"pixel_shuffle"`` (aliases:
        ``"squeeze"``, ``"zeros"``), a specific ``torch.Tensor`` or a
        ``numpy.ndarray``.
        Defaults to ``"haar"``, i.e. the `Haar transform`.
    :param resampling_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to the invertible up- and downsampling modules.
    :param disable_custom_gradient:
        If set to ``True``, `normal backpropagation` (i.e. storing
        activations instead of reconstructing activations) is used.
        Defaults to ``False``.
    :param padding_mode:
        If downsampling is not possible without residue
        (e.g. when halving spatial odd-valued resolutions), the
        input gets padded to allow for invertibility of the padded
        input. padding_mode takes the same keywords as
        ``torch.nn.functional.pad`` for ``mode``. If set to ``None``,
        this behavior is deactivated.
        Defaults to ``"constant"``.
    :param padding_value:
        If ``padding_mode`` is set to `constant`, this
        is the value that the input is padded with, e.g. 0.
        Defaults to ``0``.
    :param revert_input_padding:
        Whether to revert the input padding in the output, if desired.
        Defaults to ``True``.
    :param verbose:
        Level of verbosity. Currently only 0 (no warnings) or 1,
        which includes warnings.
        Defaults to ``1``.
    """

    def __init__(self,
                 in_channels: int,
                 architecture: Tuple[int, ...],
                 dim: int,
                 create_module_fn: Callable[[int, Optional[dict]], nn.Module]
                 = create_standard_module,
                 module_kwargs: dict = None,
                 slice_mode: str = "double",
                 learnable_resampling: bool = True,
                 resampling_stride: int = 2,
                 resampling_method: str = "cayley",
                 resampling_init: Union[str, np.ndarray, torch.Tensor] = "haar",
                 resampling_kwargs: dict = None,
                 padding_mode: Union[str, type(None)] = "constant",
                 padding_value: int = 0,
                 revert_input_padding: bool = True,
                 disable_custom_gradient: bool = False,
                 verbose: int = 1,
                 downsample = False,
                 symmetry = False,
                 **kwargs: Any):

        super(iUNet, self).__init__()
        self.downsample = downsample
        self.architecture = architecture
        self.dim = dim
        self.create_module_fn = create_module_fn
        self.disable_custom_gradient = disable_custom_gradient
        self.num_levels = len(architecture)
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = module_kwargs

        self.channels = [in_channels]
        self.channels_before_downsampling = []
        self.skipped_channels = []

        # --- Padding attributes ---
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.revert_input_padding = revert_input_padding

        # --- Invertible up- and downsampling attributes ---
        # Reformat resampling_stride to the standard format
        self.resampling_stride = self.__format_stride__(resampling_stride)
        # Calculate the channel multipliers per downsampling operation
        self.channel_multipliers = [
            int(np.prod(stride)) for stride in self.resampling_stride
        ]
        self.resampling_method = resampling_method
        self.resampling_init = resampling_init
        if resampling_kwargs is None:
            resampling_kwargs = {}
        self.resampling_kwargs = resampling_kwargs
        # Calculate the total downsampling factor per spatial dimension
        self.downsampling_factors = self.__total_downsampling_factor__(
            self.resampling_stride
        )

        # Standard behavior of self.slice_mode
        if slice_mode is "double" or slice_mode is "constant":
            if slice_mode is "double": factor = 2
            if slice_mode is "constant": factor = 1

            for i in range(len(architecture) - 1):
                self.skipped_channels.append(
                    int(
                        max([1, np.floor(
                            (self.channels[i] *
                             (self.channel_multipliers[i] - factor))
                            / self.channel_multipliers[i])]
                            )
                    )
                )
                self.channels_before_downsampling.append(
                    self.channels[i] - self.skipped_channels[-1]
                )
                self.channels.append(
                    self.channel_multipliers[i]
                    * self.channels_before_downsampling[i]
                )
        else:
            raise AttributeError(
                "Currently, only slice_mode='double' and 'constant' are "
                "supported."
            )

        # Verbosity level
        self.verbose = verbose

        # Create the architecture of the iUNet
        downsampling_op = [None,
                           None,
                           downpsi3d][dim - 1]

        upsampling_op = [None,
                         None,
                         uppsi3d][dim - 1]

        self.module_L = nn.ModuleList()
        self.module_R = nn.ModuleList()
        self.slice_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()

        self.pre_module =  nn.Conv3d(2, 16, 3, stride=2,padding=1, bias=True)

        if self.downsample == False:
            if symmetry == False:
                self.after_module =  nn.ConvTranspose3d(16, 3, 4, stride=2,padding=1, bias=False)
            else:
                self.after_module =  nn.ConvTranspose3d(16, 6, 4, stride=2,padding=1, bias=False)

        else:
            self.after_module = nn.Conv3d(8, 3, 3, stride=1, padding=1, bias=True)

        for i, num_layers in enumerate(architecture):

            current_channels = self.channels[i]

            if i < len(architecture) - 1:
                # Slice and concatenation layers
                self.slice_layers.append(
                    InvertibleModuleWrapper(
                        SplitChannels(
                            self.skipped_channels[i]
                        ),
                        disable=disable_custom_gradient
                    )
                )
                self.concat_layers.append(
                    InvertibleModuleWrapper(
                        ConcatenateChannels(
                            self.skipped_channels[i]
                        ),
                        disable=disable_custom_gradient
                    )
                )

                # Upsampling and downsampling layers
                downsampling = downsampling_op(
                    2
                )

                upsampling = upsampling_op(
                    2
                )

                # Initialize the learnable upsampling with the same
                # kernel as the learnable downsampling. This way, by
                # zero-initialization of the coupling layers, the
                # invertible U-Net is initialized as the identity
                # function.
                if learnable_resampling:
                    upsampling.kernel_matrix.data = \
                        downsampling.kernel_matrix.data

                self.downsampling_layers.append(
                    InvertibleModuleWrapper(
                        downsampling,
                        disable=learnable_resampling
                    )
                )

                self.upsampling_layers.append(
                    InvertibleModuleWrapper(
                        upsampling,
                        disable=learnable_resampling
                    )
                )

            self.module_L.append(nn.ModuleList())
            self.module_R.append(nn.ModuleList())

            for j in range(num_layers):
                coordinate_kwargs = {
                    'dim': self.dim,
                    'LR': 'L',
                    'level': i,
                    'module_index': j,
                    'architecture': self.architecture,
                }
                self.module_L[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                            self.channels[i],
                            **coordinate_kwargs,
                            **module_kwargs),
                        disable=disable_custom_gradient
                    )
                )

                coordinate_kwargs['LR'] = 'R'
                self.module_R[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                            self.channels[i],
                            **coordinate_kwargs,
                            **module_kwargs),
                        disable=disable_custom_gradient
                    )
                )

    def get_padding(self, x: torch.Tensor):
        """Calculates the required padding for the input.
        """
        shape = x.shape[2:]
        factors = self.downsampling_factors
        padded_shape = [
            int(np.ceil(s / f)) * f for (s, f) in zip(shape, factors)
        ]
        total_padding = [p - s for (s, p) in zip(shape, padded_shape)]

        # Pad evenly on all sides
        padding = [None] * (2 * len(shape))
        padding[::2] = [p - p // 2 for p in total_padding]
        padding[1::2] = [p // 2 for p in total_padding]

        # Weird thing about F.pad: While the torch data format is
        # (DHW), the padding format is (WHD).
        padding = padding[::-1]

        return padded_shape, padding

    def revert_padding(self, x: torch.Tensor, padding: List[int]):
        """Reverses a given padding.

        :param x:
            The image that was originally padded.
        :param padding:
            The padding that is removed from ``x``.
        """
        if self.dim == 1:
            x = x[:, :,
                padding[0]:-padding[1]]
        if self.dim == 2:
            x = x[:, :,
                padding[2]:-padding[3],
                padding[0]:-padding[1]]
        if self.dim == 3:
            x = x[:, :,
                padding[4]:-padding[5],
                padding[2]:-padding[3],
                padding[0]:-padding[1]]

        return x

    def __check_stride_format__(self, stride):
        """Check whether the stride has the correct format to be parsed.
        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        e.g. ``2`, ``(2,1,3)``, ``[(2,1,3), (2,2,2), (4,3,1)]``.
        """

        def raise_format_error():
            raise AttributeError(
                "resampling_stride has the wrong format. "
                "The format can be either a single integer, a single tuple "
                "(where the length corresponds to the spatial dimensions of the "
                "data), or a list containing either of the last two options "
                "(where the length of the list has to be equal to the number "
                "of downsampling operations), e.g. 2, (2,1,3), "
                "[(2,1,3), (2,2,2), (4,3,1)]. "
            )

        if isinstance(stride, int):
            pass
        elif isinstance(stride, tuple):
            if len(stride) == self.dim:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        elif isinstance(stride, list):
            if len(stride) == self.num_levels - 1:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        else:
            raise_format_error()

    def __format_stride__(self, stride):
        """Parses the resampling_stride and reformats it into a standard format.
        """
        self.__check_stride_format__(stride)
        if isinstance(stride, int):
            return [(stride,) * self.dim] * (self.num_levels - 1)
        if isinstance(stride, tuple):
            return [stride] * (self.num_levels - 1)
        if isinstance(stride, list):
            for i, element in enumerate(stride):
                if isinstance(element, int):
                    stride[i] = (element,) * self.dim
            return stride

    def __total_downsampling_factor__(self, stride):
        factors = [1] * len(stride[0])
        for i, element_tuple in enumerate(stride):
            for j, element_int in enumerate(stride[i]):
                factors[j] = factors[j] * element_int
        return tuple(factors)

    def forward(self, x: torch.Tensor):
        """Applies the forward mapping of the iUNet to ``x``.
        """
        #if not x.shape[1] == self.channels[0]:
        #    raise RuntimeError(
        #        "The number of channels does not match in_channels."
        #    )
        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warn(
                    "Input resolution {} cannot be downsampled {}  times "
                    "without residuals. Padding to resolution {} is  applied "
                    "with mode {} to retain invertibility. Set "
                    "padding_mode=None to deactivate padding. If so, expect "
                    "errors.".format(
                        list(x.shape[2:]),
                        len(self.architecture) - 1,
                        padded_shape,
                        self.padding_mode
                    )
                )

            x = nn.functional.pad(
                x, padding, self.padding_mode, self.padding_value
            )

        # skip_inputs is a list of the skip connections
        skip_inputs = []
        x = self.pre_module(x)
        # Left side
        for i in range(self.num_levels):
            depth = self.architecture[i]

            # RevNet L
            for j in range(depth):
                x = self.module_L[i][j](x)

            # Downsampling L
            if i < self.num_levels - 1:
                y, x = self.slice_layers[i](x)
                skip_inputs.append(y)
                x = self.downsampling_layers[i](x)

        # Right side
        for i in range(self.num_levels - 1, -1, -1):
            depth = self.architecture[i]

            # Upsampling R
            if i < self.num_levels - 1:
                y = skip_inputs.pop()
                x = self.upsampling_layers[i](x)
                x = self.concat_layers[i](y, x)

            # RevNet R
            for j in range(depth):
                x = self.module_R[i][j](x)
        x = self.after_module(x)

        if self.padding_mode is not None and self.revert_input_padding:
            #x = self.revert_padding(x, padding)
            pass
        return x

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the iUNet to ``x``.
        """

        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warn(
                    "Input shape to the inverse mapping requires padding."
                )
            x = nn.functional.pad(
                x, padding, self.padding_mode, self.padding_value)

        skip_inputs = []

        # Right side
        for i in range(self.num_levels):
            depth = self.architecture[i]

            # RevNet R
            for j in range(depth - 1, -1, -1):
                x = self.module_R[i][j].inverse(x)

            # Downsampling R
            if i < self.num_levels - 1:
                y, x = self.concat_layers[i].inverse(x)
                skip_inputs.append(y)
                x = self.upsampling_layers[i].inverse(x)

        # Left side
        for i in range(self.num_levels - 1, -1, -1):
            depth = self.architecture[i]

            # Upsampling L
            if i < self.num_levels - 1:
                y = skip_inputs.pop()
                x = self.downsampling_layers[i].inverse(x)
                x = self.slice_layers[i].inverse(y, x)

            # RevNet L
            for j in range(depth - 1, -1, -1):
                x = self.module_L[i][j].inverse(x)

        if self.padding_mode is not None and self.revert_input_padding:
            if self.verbose:
                warnings.warn(
                    "revert_input_padding is set to True, which may yield "
                    "non-exact reconstructions of the unpadded input."
                )
            x = self.revert_padding(x, padding)
        return x

    def print_layout(self):
        """Prints the layout of the iUNet.
        """
        print_iunet_layout(self)


class InvertibleCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, preserve_rng_state, num_inputs, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.preserve_rng_state = preserve_rng_state
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*inputs)

        ctx.input_requires_grad = [element.requires_grad for element in inputs]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = [element.detach() for element in inputs]
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_outputs = tuple([element.detach_() for element in outputs])

        # clear memory from inputs
        if not ctx.keep_input:
            if not pytorch_version_one_and_above:
                # PyTorch 0.4 way to clear storage
                for element in inputs:
                    element.data.set_()
            else:
                # PyTorch 1.0+ way to clear storage
                for element in inputs:
                    element.storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpointFunction for more than "
                               "{} times! Try raising `num_bwd_passes` by one.".format(ctx.num_bwd_passes))
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input if necessary
        if not ctx.keep_input:
            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                # recompute input
                with torch.no_grad():
                    inputs_inverted = ctx.fn_inverse(*outputs)
                    if not isinstance(inputs_inverted, tuple):
                        inputs_inverted = (inputs_inverted,)
                    if pytorch_version_one_and_above:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.storage().resize_(int(np.prod(element_original.size())))
                            element_original.set_(element_inverted)
                            # free the outputs memory since the output variable will be the reconstructed ones
                            for element in outputs:
                                element.storage().resize_(0)
                    else:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = tuple([element.detach().requires_grad_() for element in inputs])
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        gradients = torch.autograd.grad(outputs=temp_output, inputs=detached_inputs + ctx.weights, grad_outputs=grad_outputs)

        # Free the memory of temp_output
        del temp_output

        # Setting the gradients manually on the inputs and outputs (mimic backwards)
        for element, element_grad in zip(inputs, gradients[:ctx.num_inputs]):
            element.grad = element_grad

        for element, element_grad in zip(outputs, grad_outputs):
            element.grad = element_grad
        return (None, None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1,
                 disable=False, preserve_rng_state=False):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.
        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.
            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.
            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.
            disable : :obj:`bool`, optional
                This will disable using the InvertibleCheckpointFunction altogether.
                Essentially this renders the function as `y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.
            preserve_rng_state : :obj:`bool`, optional
                Setting this will ensure that the same RNG state is used during reconstruction of the inputs.
                I.e. if keep_input = False on forward or keep_input_inverse = False on inverse. By default
                this is False since most invertible modules should have a valid inverse and hence are
                deterministic.
        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.
            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.
        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self.preserve_rng_state = preserve_rng_state
        self._fn = fn

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`
        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).
        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.
        """
        if not self.disable:
            y = InvertibleCheckpointFunction.apply(
                self._fn.forward,
                self._fn.inverse,
                self.keep_input,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(xin),
                *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            y = self._fn(*xin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`
        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).
        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.
        """
        if not self.disable:
            x = InvertibleCheckpointFunction.apply(
                self._fn.inverse,
                self._fn.forward,
                self.keep_input_inverse,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(yin),
                *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            x = self._fn.inverse(*yin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x



def create_coupling(Fm, Gm=None, coupling='additive', implementation_fwd=-1, implementation_bwd=-1, adapter=None):
    if coupling == 'additive':
        fn = AdditiveCoupling(Fm, Gm,
                              implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
    else:
        raise NotImplementedError('Unknown coupling method: %s' % coupling)
    return fn


def is_invertible_module(module_in, test_input_shape, test_input_dtype=torch.float32, atol=1e-6, random_seed=42):
    """Test if a :obj:`torch.nn.Module` is invertible
    Parameters
    ----------
    module_in : :obj:`torch.nn.Module`
        A torch.nn.Module to test.
    test_input_shape : :obj:`tuple` of :obj:`int` or :obj:`tuple` of :obj:`tuple` of :obj:`int`
        Dimensions of test tensor(s) object to perform the test with.
    test_input_dtype : :obj:`torch.dtype`, optional
        Data type of test tensor object to perform the test with.
    atol : :obj:`float`, optional
        Tolerance value used for comparing the outputs.
    random_seed : :obj:`int`, optional
        Use this value to seed the pseudo-random test_input_shapes with different numbers.
    Returns
    -------
        :obj:`bool`
            True if the input module is invertible, False otherwise.
    """
    if isinstance(module_in, InvertibleModuleWrapper):
        module_in = module_in._fn

    if not hasattr(module_in, "inverse"):
        return False

    def _type_check_input_shape(test_input_shape):
        if isinstance(test_input_shape, (tuple, list)):
            if all([isinstance(e, int) for e in test_input_shape]):
                return True
            elif all([isinstance(e, (tuple, list)) for e in test_input_shape]):
                return all([isinstance(ee, int) for e in test_input_shape for ee in e])
            else:
                return False
        else:
            return False

    if not _type_check_input_shape(test_input_shape):
        raise ValueError("test_input_shape should be of type Tuple[int, ...] or "
                         "Tuple[Tuple[int, ...], ...], but {} found".format(type(test_input_shape)))

    if not isinstance(test_input_shape[0], (tuple, list)):
        test_input_shape = (test_input_shape,)

    def _check_inputs_allclose(inputs, reference, atol):
        for inp, ref in zip(inputs, reference):
            if not torch.allclose(inp, ref, atol=atol):
                return False
        return True

    def _pack_if_no_tuple(x):
        if not isinstance(x, tuple):
            return (x, )
        return x

    with torch.no_grad():
        torch.manual_seed(random_seed)
        test_inputs = tuple([torch.rand(shape, dtype=test_input_dtype) for shape in test_input_shape])
        if any([torch.equal(torch.zeros_like(e), e) for e in test_inputs]):  # pragma: no cover
            warnings.warn("Some inputs were detected to be all zeros, you might want to set a different random_seed.")

        if not _check_inputs_allclose(_pack_if_no_tuple(module_in.inverse(*_pack_if_no_tuple(module_in(*test_inputs)))), test_inputs, atol=atol):
            return False

        test_outputs = _pack_if_no_tuple(module_in(*test_inputs))
        if any([torch.equal(torch.zeros_like(e), e) for e in test_outputs]):  # pragma: no cover
            warnings.warn("Some outputs were detected to be all zeros, you might want to set a different random_seed.")

        if not _check_inputs_allclose(_pack_if_no_tuple(module_in(*_pack_if_no_tuple(module_in.inverse(*test_outputs)))), test_outputs, atol=atol):  # pragma: no cover
            return False

        test_reconstructed_inputs = _pack_if_no_tuple(module_in.inverse(*test_outputs))

    def _test_shared(inputs, outputs, msg):
        shared = set(inputs)
        shared_outputs = set(outputs)
        if len(inputs) != len(shared):  # pragma: no cover
            warnings.warn("Some inputs (*x) share the same tensor, are you sure this is what you want? ({})".format(msg))
        if len(outputs) != len(shared_outputs):
            warnings.warn("Some outputs (*y) share the same tensor, are you sure this is what you want? ({})".format(msg))
        if any([inp in shared for inp in shared_outputs]):
            warnings.warn("Some inputs (*x) and outputs (*y) share the same tensor, this is typically not a "
                          "good function to use with memcnn.InvertibleModuleWrapper as it might increase memory usage. "
                          "E.g. an identity function. ({})".format(msg))

    _test_shared(test_inputs, test_outputs, msg="forward")
    _test_shared(test_reconstructed_inputs, test_outputs, msg="inverse")

    return True


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
#
# get_device_states and set_device_states cannot be imported from torch.utils.checkpoint, since it was not
# present in older versions, so we include a copy here.
def get_device_states(*args):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


class downpsi3d(nn.Module):
    def __init__(self, block_size):
        super(downpsi3d, self).__init__()
        self.block_size = block_size
        self.block_size_qb = block_size*block_size*block_size

    def inverse(self, input):
        bl, bl_qb = self.block_size, self.block_size_qb
        bs, new_d, h, w,z = input.shape[0], input.shape[1] // bl_qb, input.shape[2], input.shape[3],input.shape[4]
        return input.reshape(bs, bl, bl,bl, new_d, h, w,z).permute(0, 4, 5, 1, 6, 2,7,3).reshape(bs, new_d, h * bl, w * bl,z*bl)

    def forward(self, input):
        bl, bl_qb = self.block_size, self.block_size_qb
        bs, d, new_h, new_w, new_z = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl, input.shape[4] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl, new_z, bl).permute(0, 3, 5, 7, 1, 2, 4, 6).reshape(bs, d * bl_qb, new_h, new_w, new_z)

class uppsi3d(nn.Module):
    def __init__(self, block_size):
        super(uppsi3d, self).__init__()
        self.block_size = block_size
        self.block_size_qb = block_size*block_size*block_size

    def forward(self, input):
        bl, bl_qb = self.block_size, self.block_size_qb
        bs, new_d, h, w,z = input.shape[0], input.shape[1] // bl_qb, input.shape[2], input.shape[3],input.shape[4]
        return input.reshape(bs, bl, bl,bl, new_d, h, w,z).permute(0, 4, 5, 1, 6, 2,7,3).reshape(bs, new_d, h * bl, w * bl,z*bl)

    def inverse(self, input):
        bl, bl_qb = self.block_size, self.block_size_qb
        bs, d, new_h, new_w, new_z = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl, input.shape[4] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl, new_z, bl).permute(0, 3, 5, 7, 1, 2, 4, 6).reshape(bs, d * bl_qb, new_h, new_w, new_z)


def set_device_states(devices, states):
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

class AdditiveCoupling(nn.Module):
    def __init__(self, Fm, Gm=None, implementation_fwd=-1, implementation_bwd=-1, split_dim=1):
        """
        This computes the output :math:`y` on forward given input :math:`x` and arbitrary modules :math:`Fm` and :math:`Gm` according to:
        :math:`(x1, x2) = x`
        :math:`y1 = x1 + Fm(x2)`
        :math:`y2 = x2 + Gm(y1)`
        :math:`y = (y1, y2)`
        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
            Gm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)
            implementation_fwd : :obj:`int`
                Switch between different Additive Operation implementations for forward pass. Default = -1
            implementation_bwd : :obj:`int`
                Switch between different Additive Operation implementations for inverse pass. Default = -1
            split_dim : :obj:`int`
                Dimension to split the input tensors on. Default = 1, generally corresponding to channels.
        """
        super(AdditiveCoupling, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.implementation_fwd = implementation_fwd
        self.implementation_bwd = implementation_bwd
        self.split_dim = split_dim
        if implementation_bwd != -1 or implementation_fwd != -1:
            warnings.warn("Other implementations than the default (-1) are now deprecated.",
                          DeprecationWarning)

    def forward(self, x):
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]

        if self.implementation_fwd == 0:
            out = AdditiveBlockFunction.apply(*args)
        elif self.implementation_fwd == 1:
            out = AdditiveBlockFunction2.apply(*args)
        elif self.implementation_fwd == -1:
            x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
            x1, x2 = x1.contiguous(), x2.contiguous()
            fmd = self.Fm.forward(x2)
            y1 = x1 + fmd
            gmd = self.Gm.forward(y1)
            y2 = x2 + gmd
            out = torch.cat([y1, y2], dim=self.split_dim)
        else:
            raise NotImplementedError("Selected implementation ({}) not implemented..."
                                      .format(self.implementation_fwd))
        return out

    def inverse(self, y):
        args = [y, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]

        if self.implementation_bwd == 0:
            x = AdditiveBlockInverseFunction.apply(*args)
        elif self.implementation_bwd == 1:
            x = AdditiveBlockInverseFunction2.apply(*args)
        elif self.implementation_bwd == -1:
            y1, y2 = torch.chunk(y, 2, dim=self.split_dim)
            y1, y2 = y1.contiguous(), y2.contiguous()
            gmd = self.Gm.forward(y1)
            x2 = y2 - gmd
            fmd = self.Fm.forward(x2)
            x1 = y1 - fmd
            x = torch.cat([x1, x2], dim=self.split_dim)
        else:
            raise NotImplementedError("Inverse for selected implementation ({}) not implemented..."
                                      .format(self.implementation_bwd))
        return x


class AdditiveBlock(AdditiveCoupling):
    def __init__(self, Fm, Gm=None, implementation_fwd=1, implementation_bwd=1):
        warnings.warn("This class has been deprecated. Use the AdditiveCoupling class instead.",
                      DeprecationWarning)
        super(AdditiveBlock, self).__init__(Fm=Fm, Gm=Gm,
                                            implementation_fwd=implementation_fwd,
                                            implementation_bwd=implementation_bwd)


class AdditiveBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xin, Fm, Gm, *weights):
        """Forward pass computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}
        Parameters
        ----------
        ctx : torch.autograd.Function
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this function
        """
        # check if possible to partition into two equally sized partitions
        assert(xin.shape[1] % 2 == 0)  # nosec

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        with torch.no_grad():
            x = xin.detach()
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            fmr = Fm.forward(x2)

            y1 = x1 + fmr
            x1.set_()
            del x1
            gmr = Gm.forward(y1)
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)

        ctx.save_for_backward(xin, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        # retrieve weight references
        Fm, Gm = ctx.Fm, ctx.Gm

        # retrieve input and output references
        xin, output = ctx.saved_tensors
        x = xin.detach()
        x1, x2 = torch.chunk(x, 2, dim=1)
        GWeights = [p for p in Gm.parameters()]
        # partition output gradient also on channels
        assert grad_output.shape[1] % 2 == 0  # nosec

        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1.requires_grad_()
            x2.requires_grad_()

            y1 = x1 + Fm.forward(x2)
            y2 = x2 + Gm.forward(y1)
            y = torch.cat([y1, y2], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(y, (x1, x2 ) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output)

            GWgrads = dd[2:2+len(GWeights)]
            FWgrads = dd[2+len(GWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

        return (grad_input, None, None) + FWgrads + GWgrads


class AdditiveBlockInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass computes:
        {y1, y2} = y
        x2 = y2 - Gm(y1)
        x1 = y1 - Fm(x2)
        output = {x1, x2}
        Parameters
        ----------
        cty : torch.autograd.Function
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction
        """
        # check if possible to partition into two equally sized partitions
        assert(y.shape[1] % 2 == 0)  # nosec

        # store partition size, Fm and Gm functions in context
        cty.Fm = Fm
        cty.Gm = Gm

        with torch.no_grad():
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs
            gmr = Gm.forward(y1)

            x2 = y2 - gmr
            y2.set_()
            del y2
            fmr = Fm.forward(x2)
            x1 = y1 - fmr
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1)
            x1.set_()
            x2.set_()
            del x1, x2

        # save the (empty) input and (non-empty) output variables
        cty.save_for_backward(y.data, output)

        return output

    @staticmethod
    def backward(cty, grad_output):  # pragma: no cover
        # retrieve weight references
        Fm, Gm = cty.Fm, cty.Gm

        # retrieve input and output references
        yin, output = cty.saved_tensors
        y = yin.detach()
        y1, y2 = torch.chunk(y, 2, dim=1)
        FWeights = [p for p in Fm.parameters()]

        # partition output gradient also on channels
        assert grad_output.shape[1] % 2 == 0  # nosec

        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            y2.requires_grad = True
            y1.requires_grad = True

            x2 = y2 - Gm.forward(y1)
            x1 = y1 - Fm.forward(x2)
            x = torch.cat([x1, x2], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(x, (y2, y1 ) + tuple(Fm.parameters()) + tuple(Gm.parameters()), grad_output)

            FWgrads = dd[2:2+len(FWeights)]
            GWgrads = dd[2+len(FWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

        return (grad_input, None, None) + FWgrads + GWgrads

class AdditiveBlockFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xin, Fm, Gm, *weights):
        """Forward pass computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}
        Parameters
        ----------
        ctx : torch.autograd.Function
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction
        """
        # check if possible to partition into two equally sized partitions
        assert xin.shape[1] % 2 == 0  # nosec

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        with torch.no_grad():
            # partition in two equally sized set of channels
            x = xin.detach()
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            fmr = Fm.forward(x2)

            y1 = x1 + fmr
            x1.set_()
            del x1
            gmr = Gm.forward(y1)
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1).detach_()

        # save the input and output variables
        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        x, output = ctx.saved_tensors

        with torch.no_grad():
            y1, y2 = torch.chunk(output, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.shape[1] % 2 == 0)  # nosec
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)
            y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        with set_grad_enabled(True):
            z1_stop = y1.detach()
            z1_stop.requires_grad = True

            G_z1 = Gm.forward(z1_stop)
            x2 = y2 - G_z1
            x2_stop = x2.detach()
            x2_stop.requires_grad = True

            F_x2 = Fm.forward(x2_stop)
            x1 = y1 - F_x2
            x1_stop = x1.detach()
            x1_stop.requires_grad = True

            # compute outputs building a sub-graph
            y1 = x1_stop + F_x2
            y2 = x2_stop + G_z1

            # calculate the final gradients for the weights and inputs
            dd = torch.autograd.grad(y2, (z1_stop,) + tuple(Gm.parameters()), y2_grad, retain_graph=False)
            z1_grad = dd[0] + y1_grad
            GWgrads = dd[1:]

            dd = torch.autograd.grad(y1, (x1_stop, x2_stop) + tuple(Fm.parameters()), z1_grad, retain_graph=False)

            FWgrads = dd[2:]
            x2_grad = dd[1] + y2_grad
            x1_grad = dd[0]
            grad_input = torch.cat([x1_grad, x2_grad], dim=1)

        return (grad_input, None, None) + FWgrads + GWgrads


class AdditiveBlockInverseFunction2(torch.autograd.Function):
    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass computes:
        {y1, y2} = y
        x2 = y2 - Gm(y1)
        x1 = y1 - Fm(x2)
        output = {x1, x2}
        Parameters
        ----------
        cty : torch.autograd.Function
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction
        """
        # check if possible to partition into two equally sized partitions
        assert(y.shape[1] % 2 == 0)  # nosec

        # store partition size, Fm and Gm functions in context
        cty.Fm = Fm
        cty.Gm = Gm

        with torch.no_grad():
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs
            gmr = Gm.forward(y1)

            x2 = y2 - gmr
            y2.set_()
            del y2
            fmr = Fm.forward(x2)
            x1 = y1 - fmr
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1).detach_()

        # save the input and output variables
        cty.save_for_backward(y, output)

        return output

    @staticmethod
    def backward(cty, grad_output):  # pragma: no cover

        Fm, Gm = cty.Fm, cty.Gm
        # are all variable objects now
        y, output = cty.saved_tensors

        with torch.no_grad():
            x1, x2 = torch.chunk(output, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.shape[1] % 2 == 0)  # nosec
            x1_grad, x2_grad = torch.chunk(grad_output, 2, dim=1)
            x1_grad, x2_grad = x1_grad.contiguous(), x2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, y1_stop, GW, FW
        # Also recompute inputs (y1, y2) from outputs (x1, x2)
        with set_grad_enabled(True):
            z1_stop = x2.detach()
            z1_stop.requires_grad = True

            F_z1 = Fm.forward(z1_stop)
            y1 = x1 + F_z1
            y1_stop = y1.detach()
            y1_stop.requires_grad = True

            G_y1 = Gm.forward(y1_stop)
            y2 = x2 + G_y1
            y2_stop = y2.detach()
            y2_stop.requires_grad = True

            # compute outputs building a sub-graph
            z1 = y2_stop - G_y1
            x1 = y1_stop - F_z1
            x2 = z1

            # calculate the final gradients for the weights and inputs
            dd = torch.autograd.grad(x1, (z1_stop,) + tuple(Fm.parameters()), x1_grad)
            z1_grad = dd[0] + x2_grad
            FWgrads = dd[1:]

            dd = torch.autograd.grad(x2, (y2_stop, y1_stop) + tuple(Gm.parameters()), z1_grad, retain_graph=False)

            GWgrads = dd[2:]
            y1_grad = dd[1] + x1_grad
            y2_grad = dd[0]

            grad_input = torch.cat([y1_grad, y2_grad], dim=1)

        return (grad_input, None, None) + FWgrads + GWgrads

class AffineAdapterNaive(nn.Module):
    """ Naive Affine adapter
        Outputs exp(f(x)), f(x) given f(.) and x
    """
    def __init__(self, module):
        super(AffineAdapterNaive, self).__init__()
        self.f = module

    def forward(self, x):
        t = self.f(x)
        s = torch.exp(t)
        return s, t


def __calculate_kernel_matrix_exp__(weight, *args, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return expm.apply(skew_symmetric_matrix)


def __calculate_kernel_matrix_cayley__(weight, *args, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return cayley.apply(skew_symmetric_matrix)


def __calculate_kernel_matrix_householder__(weight, *args, **kwargs):
    raise NotImplementedError("Parametrization via Householder transform "
                              "not implemented.")


def __calculate_kernel_matrix_givens__(weight, *args, **kwargs):
    raise NotImplementedError("Parametrization via Givens rotations not "
                              "implemented.")


def __calculate_kernel_matrix_bjork__(weight, *args, **kwargs):
    raise NotImplementedError("Parametrization via Bjork peojections "
                              "not implemented.")


class OrthogonalResamplingLayer(torch.nn.Module):
    """Base class for orthogonal up- and downsampling operators.
    :param low_channel_number:
        Lower number of channels. These are the input
        channels in the case of downsampling ops, and the output
        channels in the case of upsampling ops.
    :param stride:
        The downsampling / upsampling factor for each dimension.
    :param channel_multiplier:
        The channel multiplier, i.e. the number
        by which the number of channels are multiplied (downsampling)
        or divided (upsampling).
    :param method:
        Which method to use for parametrizing orthogonal
        matrices which are used as convolutional kernels.
    """

    def __init__(self,
                 low_channel_number: int,
                 stride: Union[int, Tuple[int, ...]],
                 method: str = 'cayley',
                 init: Union[str, np.ndarray, torch.Tensor] = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):

        super(OrthogonalResamplingLayer, self).__init__()
        self.low_channel_number = low_channel_number
        self.method = method
        self.stride = stride
        self.channel_multiplier = int(np.prod(stride))
        self.high_channel_number = self.channel_multiplier * low_channel_number

        assert (method in ['exp', 'cayley'])
        if method is 'exp':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_exp__
        elif method is 'cayley':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_cayley__

        self._kernel_matrix_shape = ((self.low_channel_number,)
                                     + (self.channel_multiplier,) * 2)
        self._kernel_shape = ((self.high_channel_number, 1)
                              + self.stride)

        self.weight = torch.nn.Parameter(
            __initialize_weight__(kernel_matrix_shape=self._kernel_matrix_shape,
                                  stride=self.stride,
                                  method=self.method,
                                  init=init)
        )
        self.weight.requires_grad = learnable

    # Apply the chosen method to the weight in order to parametrize
    # an orthogonal matrix, then reshape into a convolutional kernel.
    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight)

    @property
    def kernel(self):
        """The kernel associated with the invertible up-/downsampling.
        """
        return self.kernel_matrix.reshape(*self._kernel_shape)


class InvertibleDownsampling3D(OrthogonalResamplingLayer):
    def __init__(self,
                 in_channels: int,
                 stride = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = tuple(_triple(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling3D, self).__init__(
            low_channel_number=self.in_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
            *args,
            **kwargs
        )

    def forward(self, x):
        # Convolve with stride 2 in order to invertibly downsample.
        return F.conv3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        # Apply transposed convolution in order to invert the downsampling.
        return F.conv_transpose3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class InvertibleUpsampling3D(OrthogonalResamplingLayer):
    def __init__(self,
                 in_channels: int,
                 stride = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = tuple(_triple(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling3D, self).__init__(
            low_channel_number=self.out_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
            *args,
            **kwargs
        )

    def forward(self, x):
        # Apply transposed convolution in order to invertibly upsample.
        return F.conv_transpose3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        # Convolve with stride 2 in order to invert the upsampling.
        return F.conv3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class SplitChannels(torch.nn.Module):
    def __init__(self, split_location):
        super(SplitChannels, self).__init__()
        self.split_location = split_location

    def forward(self, x):
        a, b = (x[:, :self.split_location],
                x[:, self.split_location:])
        a, b = a.clone(), b.clone()
        del x
        return a, b

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


class ConcatenateChannels(torch.nn.Module):
    def __init__(self, split_location):
        super(ConcatenateChannels, self).__init__()
        self.split_location = split_location

    def forward(self, x, y):
        return torch.cat([x, y], dim=1)

    def inverse(self, x):
        a, b = (x[:, :self.split_location],
                x[:, self.split_location:])
        a, b = a.clone(), b.clone()
        del x
        return a, b


class StandardAdditiveCoupling(nn.Module):
    """
    This computes the output :math:`y` on forward given input :math:`x`
    and arbitrary modules :math:`F` according to:
    :math:`(x1, x2) = x`
    :math:`y1 = x2`
    :math:`y2 = x1 + F(y2)`
    :math:`y = (y1, y2)`
    Parameters
    ----------
        Fm : :obj:`torch.nn.Module`
            A torch.nn.Module encapsulating an arbitrary function
    """

    def __init__(self, F, channel_split_pos):
        super(StandardAdditiveCoupling, self).__init__()
        self.F = F
        self.channel_split_pos = channel_split_pos

    def forward(self, x):
        x1, x2 = x[:, :self.channel_split_pos], x[:, self.channel_split_pos:]
        x1, x2 = x1.contiguous(), x2.contiguous()
        y1 = x2
        y2 = x1 + self.F.forward(x2)
        out = torch.cat([y1, y2], dim=1)
        return out

    def inverse(self, y):
        # y1, y2 = torch.chunk(y, 2, dim=1)
        inverse_channel_split_pos = y.shape[1] - self.channel_split_pos
        y1, y2 = y[:, :inverse_channel_split_pos], y[:, inverse_channel_split_pos:]
        y1, y2 = y1.contiguous(), y2.contiguous()
        x2 = y1
        x1 = y2 - self.F.forward(y1)
        x = torch.cat([x1, x2], dim=1)
        return x


class StandardBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_in_channels,
                 num_out_channels,
                 block_depth=1,
                 zero_init=True):
        super(StandardBlock, self).__init__()

        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]

        self.seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        for i in range(block_depth):

            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)

            if i == 0:
                current_in_channels = num_in_channels
            if i == block_depth - 1:
                current_out_channels = num_out_channels

            self.seq.append(
                conv_op(
                    current_in_channels,
                    current_out_channels,
                    3,
                    padding=1,
                    bias=False))
            torch.nn.init.kaiming_uniform_(self.seq[-1].weight,
                                           a=0.01,
                                           mode='fan_out',
                                           nonlinearity='leaky_relu')

            self.seq.append(nn.LeakyReLU(inplace=True))

            # With groups=1, group normalization becomes layer normalization
            self.seq.append(nn.GroupNorm(1, current_out_channels, eps=1e-3))

        # Initialize the block as the zero transform, such that the coupling
        # becomes the coupling becomes an identity transform (up to permutation
        # of channels)
        if zero_init:
            torch.nn.init.zeros_(self.seq[-1].weight)
            torch.nn.init.zeros_(self.seq[-1].bias)

        self.F = nn.Sequential(*self.seq)

    def forward(self, x):
        x = self.F(x)
        return x




def __initialize_weight__(kernel_matrix_shape: Tuple[int, ...],
                          stride: Tuple[int, ...],
                          method: str = 'cayley',
                          init: str = 'haar',
                          dtype: str = 'float32',
                          *args,
                          **kwargs):
    """Function which computes specific orthogonal matrices.
    For some chosen method of parametrizing orthogonal matrices, this
    function outputs the required weights necessary to represent a
    chosen initialization as a Pytorch tensor of matrices.
    Args:
        kernel_matrix_shape : The output shape of the
            orthogonal matrices. Should be (num_matrices, height, width).
        stride : The stride for the invertible up- or
            downsampling for which this matrix is to be used. The length
            of ``stride`` should match the dimensionality of the data.
        method : The method for parametrising orthogonal matrices.
            Should be 'exp' or 'cayley'
        init : The matrix which should be represented. Should be
            'squeeze', 'pixel_shuffle', 'haar' or 'random'. 'haar' is only
            possible if ``stride`` is only 2.
        dtype : Numpy dtype which should be used for the matrix.
        *args: Variable length argument iterable.
        **kwargs: Arbitrary keyword arguments.
    Returns:
        Tensor : Orthogonal matrices of shape ``kernel_matrix_shape``.
    """

    # Determine dimensionality of the data and the number of matrices.
    dim = len(stride)
    num_matrices = kernel_matrix_shape[0]

    # tbd: Givens, Householder, Bjork, give proper exception.
    assert (method in ['exp', 'cayley'])

    if init is 'random':
        return torch.randn(kernel_matrix_shape)

    if init is 'haar' and set(stride) != {2}:
        print("Initialization 'haar' only available for stride 2.")
        print("Falling back to 'squeeze' transform...")
        init = 'squeeze'

    if init is 'haar' and set(stride) == {2}:
        if method == 'exp':
            # The following matrices each parametrize the Haar transform when
            # exponentiating the skew symmetric matrix weight-weight.T
            # Can be derived from the theory in Gallier, Jean, and Dianna Xu.
            # "Computing exponentials of skew-symmetric matrices and logarithms
            # of orthogonal matrices." International Journal of Robotics and
            # Automation 18.1 (2003): 10-20.
            p = np.pi / 4
            if dim == 1:
                weight = np.array([[[0, p],
                                    [0, 0]]],
                                  dtype=dtype)
            if dim == 2:
                weight = np.array([[[0, 0, p, p],
                                    [0, 0, -p, -p],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]]],
                                  dtype=dtype)
            if dim == 3:
                weight = np.array(
                    [[[0, p, p, 0, p, 0, 0, 0],
                      [0, 0, 0, p, 0, p, 0, 0],
                      [0, 0, 0, p, 0, 0, p, 0],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, p, p, 0],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, 0, 0, 0]]],
                    dtype=dtype)

            return torch.tensor(weight).repeat(num_matrices, 1, 1)

        elif method is 'cayley':
            # The following matrices parametrize a Haar kernel matrix
            # when applying the Cayley transform. These can be found by
            # applying an inverse Cayley transform to a Haar kernel matrix.
            if dim == 1:
                p = -np.sqrt(2) / (2 - np.sqrt(2))
                weight = np.array([[[0, p],
                                    [0, 0]]],
                                  dtype=dtype)
            if dim == 2:
                p = .5
                weight = np.array([[[0, 0, p, p],
                                    [0, 0, -p, -p],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]]],
                                  dtype=dtype)
            if dim == 3:
                p = 1 / np.sqrt(2)
                weight = np.array(
                    [[[0, -p, -p, 0, -p, 0, 0, 1 - p],
                      [0, 0, 0, -p, 0, -p, p - 1, 0],
                      [0, 0, 0, -p, 0, p - 1, -p, 0],
                      [0, 0, 0, 0, 1 - p, 0, 0, -p],
                      [0, 0, 0, 0, 0, -p, -p, 0],
                      [0, 0, 0, 0, 0, 0, 0, -p],
                      [0, 0, 0, 0, 0, 0, 0, -p],
                      [0, 0, 0, 0, 0, 0, 0, 0]]],
                    dtype=dtype)
            return torch.tensor(weight).repeat(num_matrices, 1, 1)

    if init in ['squeeze', 'pixel_shuffle', 'zeros']:
        if method == 'exp' or method == 'cayley':
            return torch.zeros(*kernel_matrix_shape)

    # An initialization of the weight can also be explicitly provided as a
    # numpy or torch tensor. If only one matrix is provided, this matrix
    # is copied num_matrices times.
    if type(init) is np.ndarray:
        init = torch.tensor(init.astype(dtype))

    if torch.is_tensor(init):
        if len(init.shape) == 2:
            init = init.reshape(1, *init.shape)
        if init.shape[0] == 1:
            init = init.repeat(num_matrices, 1, 1)
        assert (init.shape == kernel_matrix_shape)
        return init

    else:
        raise NotImplementedError("Unknown initialization.")


class OrthogonalChannelMixing(nn.Module):
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True):
        super(OrthogonalResamplingLayer, self).__init__()

        self.in_channels = in_channels
        self.weight = nn.Parameter(
            (in_channels, in_channels),
            requires_grad=learnable
        )

        assert (method in ['exp', 'cayley'])
        if method is 'exp':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_exp__
        elif method is 'cayley':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_cayley__

    # Apply the chosen method to the weight in order to parametrize
    # an orthogonal matrix, then reshape into a convolutional kernel.
    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight)

    @property
    def kernel_matrix_transposed(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return torch.transpose(self.kernel_matrix, -1, -2)


class InvertibleChannelMixing2D(OrthogonalChannelMixing):
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True):
        super(InvertibleChannelMixing2D, self).__init__(
            in_channels=in_channels,
            method=method,
            learnable=learnable
        )

    @property
    def kernel(self):
        return self.kernel_matrix.view(
            self.in_channels, self.in_channels, 1
        )

    def forward(self, x):
        return nn.functional.conv1d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose1d(x, self.kernel)


class InvertibleChannelMixing2D(OrthogonalChannelMixing):
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True):
        super(InvertibleChannelMixing2D, self).__init__(
            in_channels=in_channels,
            method=method,
            learnable=learnable
        )

    @property
    def kernel(self):
        return self.kernel_matrix.view(
            self.in_channels, self.in_channels, 1, 1
        )

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose2d(x, self.kernel)


class InvertibleChannelMixing3D(OrthogonalChannelMixing):
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True):
        super(InvertibleChannelMixing3D, self).__init__(
            in_channels=in_channels,
            method=method,
            learnable=learnable
        )

    @property
    def kernel(self):
        return self.kernel_matrix.view(
            self.in_channels, self.in_channels, 1, 1, 1
        )

    def forward(self, x):
        return nn.functional.conv3d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose3d(x, self.kernel)


def print_iunet_layout(iunet):
    left = []
    right = []
    splits = []

    middle_padding = [''] * (iunet.num_levels)

    output = [''] * (iunet.num_levels)

    for i in range(iunet.num_levels):
        left.append(
            '-'.join([str(iunet.channels[i])] * iunet.architecture[i])
        )
        if i < iunet.num_levels - 1:
            splits.append(
                '({}/{})'.format(
                    iunet.skipped_channels[i],
                    iunet.channels_before_downsampling[i]
                )
            )
        else:
            splits.append('')
        right.append(splits[-1] + '-' + left[-1])
        left[-1] = left[-1] + '-' + splits[-1]

    for i in range(iunet.num_levels - 1, -1, -1):
        if i < iunet.num_levels - 1:
            middle_padding[i] = \
                ''.join(['-'] * max([len(output[i + 1]) - len(splits[i]), 4]))
        output[i] = left[i] + middle_padding[i] + right[i]

    for i in range(iunet.num_levels):
        if i > 0:
            outside_padding = len(output[0]) - len(output[i])
            _left = outside_padding // 2
            left_padding = ''.join(['-'] * _left)
            _right = outside_padding - _left
            right_padding = ''.join(['-'] * _right)
            output[i] = ''.join([left_padding, output[i], right_padding])
        print(output[i])


def get_num_channels(input_shape_or_channels):
    """
    Small helper function which outputs the number of
    channels regardless of whether the input shape or
    the number of channels were passed.
    """
    if hasattr(input_shape_or_channels, '__iter__'):
        return input_shape_or_channels[0]
    else:
        return input_shape_or_channels

def _cayley(A):
    I = torch.eye(A.shape[-1], device=A.device)
    LU  = torch.lu(I+A, pivot=True)
    return torch.lu_solve(I-A,*LU)

def _cayley_inverse(Q):
    I = torch.eye(Q.shape[-1], device=Q.device)
    rec_LU = torch.lu(I+Q, pivot=True)
    return torch.lu_solve(I-Q,*rec_LU)

def _cayley_frechet(A,H,Q=None):
    I = torch.eye(A.shape[-1], device=A.device)
    if Q is None:
        Q = _cayley(A)
    _LU = torch.lu(I+A, pivot=True)
    p = torch.lu_solve(Q, *_LU)
    _LU = torch.lu(I-A, pivot=True)
    q = torch.lu_solve(H, *_LU)
    return 2.* q @ p

class cayley(Function):
    """Computes the Cayley transform.
    """
    @staticmethod
    def forward(ctx, M):
        cayley_M = _cayley(M)
        ctx.save_for_backward(M, cayley_M)
        return cayley_M

    @staticmethod
    def backward(ctx, grad_out):
        M, cayley_M = ctx.saved_tensors
        dcayley_M = _cayley_frechet(M, grad_out, Q=cayley_M)
        return dcayley_M


def _eye_like(M, device=None, dtype=None):
    """Creates an identity matrix of the same shape as another matrix.
    For matrix M, the output is same shape as M, if M is a (n,n)-matrix.
    If M is a batch of m matrices (i.e. a (m,n,n)-tensor), create a batch of
    (n,n)-identity-matrices.
    Args:
        M (torch.Tensor) : A tensor of either shape (n,n) or (m,n,n), for
            which either an identity matrix or a batch of identity matrices
            of the same shape will be created.
        device (torch.device, optional) : The device on which the output
            will be placed. By default, it is placed on the same device
            as M.
        dtype (torch.dtype, optional) : The dtype of the output. By default,
            it is the same dtype as M.
    Returns:
        torch.Tensor : Identity matrix or batch of identity matrices.
    """
    assert (len(M.shape) in [2, 3])
    assert (M.shape[-1] == M.shape[-2])
    n = M.shape[-1]
    if device is None:
        device = M.device
    if dtype is None:
        dtype = M.dtype
    eye = torch.eye(M.shape[-1], device=device, dtype=dtype)
    if len(M.shape) == 2:
        return eye
    else:
        m = M.shape[0]
        return eye.view(-1, n, n).expand(m, -1, -1)


def matrix_1_norm(A):
    """Calculates the 1-norm of a matrix or a batch of matrices.
    Args:
        A (torch.Tensor): Can be either of size (n,n) or (m,n,n).
    Returns:
        torch.Tensor : The 1-norm of A.
    """
    norm, indices = torch.max(
        torch.sum(torch.abs(A), axis=-2),
        axis=-1)
    return norm


def _compute_scales(A):
    """Compute optimal parameters for scaling-and-squaring algorithm.
    The constants used in this function are determined by the MATLAB
    function found in
    https://github.com/cetmann/pytorch_expm/blob/master/determine_frechet_scaling_constant.m
    """
    norm = matrix_1_norm(A)
    max_norm = torch.max(norm)
    s = torch.zeros_like(norm)

    if A.dtype == torch.float64:
        if A.requires_grad:
            ell = {3: 0.010813385777848,
                   5: 0.199806320697895,
                   7: 0.783460847296204,
                   9: 1.782448623969279,
                   13: 4.740307543765127}
        else:
            ell = {3: 0.014955852179582,
                   5: 0.253939833006323,
                   7: 0.950417899616293,
                   9: 2.097847961257068,
                   13: 5.371920351148152}
        if max_norm >= ell[9]:
            m = 13
            magic_number = ell[m]
            s = torch.relu_(torch.ceil(torch.log2_(norm / magic_number)))
        else:
            for m in [3, 5, 7, 9]:
                if max_norm < ell[m]:
                    magic_number = ell[m]
                    # results in s = torch.tensor([0,...,0])
                    break

    elif A.dtype == torch.float32:
        if A.requires_grad:
            ell = {3: 0.308033041845330,
                   5: 1.482532614793145,
                   7: 3.248671755200478}
        else:
            ell = {3: 0.425873001692283,
                   5: 1.880152677804762,
                   7: 3.925724783138660}
        if max_norm >= ell[5]:
            m = 7
            magic_number = ell[m]
            s = torch.relu_(torch.ceil(torch.log2_(norm / magic_number)))
        else:
            for m in [3, 5]:
                if max_norm < ell[m]:
                    # results in s = torch.tensor([0,...,0])
                    magic_number = ell[m]
                    break
    return s, m


def _square(s, R, L=None):
    """The `squaring` part of the `scaling-and-squaring` algorithm.
    This works both for the forward as well as the derivative of
    the matrix exponential.
    """
    s_max = torch.max(s).int()
    if s_max > 0:
        I = _eye_like(R)
        if L is not None:
            O = torch.zeros_like(R)
        indices = [1 for k in range(len(R.shape) - 1)]

    for i in range(s_max):
        # Multiply j-th matrix by dummy identity matrices if s<[j] < s_max,
        # to prevent squaring more often than desired.
        # temp = torch.clone(R)
        mask = (i >= s)
        matrices_mask = mask.view(-1, *indices)

        # L <- R@L+L@R.
        # R <- R@R
        # If the matrices in the matrix batch require a different number
        # of squarings, individually replace matrices by identity matrices
        # in the first multiplication and by zero-matrices in the second
        # multiplication, which results in L <- L, R <- R
        temp_eye = torch.clone(R).masked_scatter(matrices_mask, I)
        if L is not None:
            temp_zeros = torch.clone(R).masked_scatter(matrices_mask, O)
            L = temp_eye @ L + temp_zeros @ L
        R = R @ temp_eye
        del temp_eye, mask

    if L is not None:
        return R, L
    else:
        return R


def _expm_scaling_squaring(A):
    """Scaling-and-squaring algorithm for matrix eponentiation.
    This is based on the observation that exp(A) = exp(A/k)^k, where
    e.g. k=2^s. The exponential exp(A/(2^s)) is calculated by a diagonal
    Padé approximation, where s is chosen based on the 1-norm of A, such
    that certain approximation guarantees can be given. exp(A) is then
    calculated by repeated squaring via exp(A/(2^s))^(2^s). This function
    works both for (n,n)-tensors as well as batchwise for (m,n,n)-tensors.
    """

    # Is A just a n-by-n matrix or does it have an extra batch dimension?
    assert (A.shape[-1] == A.shape[-2] and len(A.shape) in [2, 3])
    has_batch_dim = True if len(A.shape) == 3 else False

    # Step 1: Scale matrices in A according to a norm criterion
    s, m = _compute_scales(A)
    if torch.max(s) > 0:
        indices = [1 for k in range(len(A.shape) - 1)]
        A = A * torch.pow(2, -s).view(-1, *indices)

    # Step 2: Calculate the exponential of the scaled matrices via diagonal
    # Padé approximation.
    exp_A = _expm_pade(A, m)

    # Step 3: Square the matrices an appropriate number of times to revert
    # the scaling in step 1.
    exp_A = _square(s, exp_A)

    return exp_A


def _expm_frechet_scaling_squaring(A, E, adjoint=False):
    """Numerical Fréchet derivative of matrix exponentiation.
    """

    # Is A just a n-by-n matrix or does it have an extra batch dimension?
    assert (A.shape[-1] == A.shape[-2] and len(A.shape) in [2, 3])
    has_batch_dim = True if len(A.shape) == 3 else False

    if adjoint == True:
        A = torch.transpose(A, -1, -2)

        # Step 1: Scale matrices in A and E according to a norm criterion
    s, m = _compute_scales(A)
    if torch.max(s) > 0:
        indices = [1 for k in range(len(A.shape) - 1)]
        scaling_factors = torch.pow(2, -s).view(-1, *indices)
        A = A * scaling_factors
        E = E * scaling_factors

    # Step 2: Calculate the exponential of the scaled matrices via diagonal
    # Padé approximation, both for the exponential and its derivative.
    exp_A, dexp_A = _expm_frechet_pade(A, E, m)

    # Step 3: Square the matrices an appropriate number of times to revert
    # the scaling in step 1.
    exp_A, dexp_A = _square(s, exp_A, dexp_A)

    return dexp_A


def _expm_pade(A, m=7):
    assert (m in [3, 5, 7, 9, 13])

    # The following are values generated as
    # b = torch.tensor([_fraction(m, k) for k in range(m+1)]),
    # but reduced to natural numbers such that b[-1]=1. This still works,
    # because the same constants are used in the numerator and denominator
    # of the Padé approximation.
    if m == 3:
        b = [120., 60., 12., 1.]
    elif m == 5:
        b = [30240., 15120., 3360., 420., 30., 1.]
    elif m == 7:
        b = [17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.]
    elif m == 9:
        b = [17643225600., 8821612800., 2075673600., 302702400., 30270240.,
             2162160., 110880., 3960., 90., 1.]
    elif m == 13:
        b = [64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
             129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
             40840800., 960960., 16380., 182., 1.]

    # pre-computing terms
    I = _eye_like(A)
    if m != 13:  # There is a more efficient algorithm for m=13
        U = b[1] * I
        V = b[0] * I
        if m >= 3:
            A_2 = A @ A
            U = U + b[3] * A_2
            V = V + b[2] * A_2
        if m >= 5:
            A_4 = A_2 @ A_2
            U = U + b[5] * A_4
            V = V + b[4] * A_4
        if m >= 7:
            A_6 = A_4 @ A_2
            U = U + b[7] * A_6
            V = V + b[6] * A_6
        if m == 9:
            A_8 = A_4 @ A_4
            U = U + b[9] * A_8
            V = V + b[8] * A_8
        U = A @ U
    else:
        A_2 = A @ A
        A_4 = A_2 @ A_2
        A_6 = A_4 @ A_2

        W_1 = b[13] * A_6 + b[11] * A_4 + b[9] * A_2
        W_2 = b[7] * A_6 + b[5] * A_4 + b[3] * A_2 + b[1] * I
        W = A_6 @ W_1 + W_2

        Z_1 = b[12] * A_6 + b[10] * A_4 + b[8] * A_2
        Z_2 = b[6] * A_6 + b[4] * A_4 + b[2] * A_2 + b[0] * I

        U = A @ W
        V = A_6 @ Z_1 + Z_2

    del A_2
    if m >= 5: del A_4
    if m >= 7: del A_6
    if m == 9: del A_8

    R = torch.lu_solve(U + V, *torch.lu(-U + V))

    del U, V
    return R


def _expm_frechet_pade(A, E, m=7):
    assert (m in [3, 5, 7, 9, 13])

    if m == 3:
        b = [120., 60., 12., 1.]
    elif m == 5:
        b = [30240., 15120., 3360., 420., 30., 1.]
    elif m == 7:
        b = [17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.]
    elif m == 9:
        b = [17643225600., 8821612800., 2075673600., 302702400., 30270240.,
             2162160., 110880., 3960., 90., 1.]
    elif m == 13:
        b = [64764752532480000., 32382376266240000., 7771770303897600.,
             1187353796428800., 129060195264000., 10559470521600.,
             670442572800., 33522128640., 1323241920., 40840800., 960960.,
             16380., 182., 1.]

    # Efficiently compute series terms of M_i (and A_i if needed).
    # Not very pretty, but more readable than the alternatives.
    I = _eye_like(A)
    if m != 13:
        if m >= 3:
            M_2 = A @ E + E @ A
            A_2 = A @ A
            U = b[3] * A_2
            V = b[2] * A_2
            L_U = b[3] * M_2
            L_V = b[2] * M_2
        if m >= 5:
            M_4 = A_2 @ M_2 + M_2 @ A_2
            A_4 = A_2 @ A_2
            U = U + b[5] * A_4
            V = V + b[4] * A_4
            L_U = L_U + b[5] * M_4
            L_V = L_V + b[4] * M_4
        if m >= 7:
            M_6 = A_4 @ M_2 + M_4 @ A_2
            A_6 = A_4 @ A_2
            U = U + b[7] * A_6
            V = V + b[6] * A_6
            L_U = L_U + b[7] * M_6
            L_V = L_V + b[6] * M_6
        if m == 9:
            M_8 = A_4 @ M_4 + M_4 @ A_4
            A_8 = A_4 @ A_4
            U = U + b[9] * A_8
            V = V + b[8] * A_8
            L_U = L_U + b[9] * M_8
            L_V = L_V + b[8] * M_8

        U = U + b[1] * I
        V = U + b[0] * I
        del I

        L_U = A @ L_U
        L_U = L_U + E @ U

        U = A @ U

    else:
        M_2 = A @ E + E @ A
        A_2 = A @ A

        M_4 = A_2 @ M_2 + M_2 @ A_2
        A_4 = A_2 @ A_2

        M_6 = A_4 @ M_2 + M_4 @ A_2
        A_6 = A_4 @ A_2

        W_1 = b[13] * A_6 + b[11] * A_4 + b[9] * A_2
        W_2 = b[7] * A_6 + b[5] * A_4 + b[3] * A_2 + b[1] * I

        W = A_6 @ W_1 + W_2

        Z_1 = b[12] * A_6 + b[10] * A_4 + b[8] * A_2
        Z_2 = b[6] * A_6 + b[4] * A_4 + b[2] * A_2 + b[0] * I

        U = A @ W
        V = A_6 @ Z_1 + Z_2

        L_W1 = b[13] * M_6 + b[11] * M_4 + b[9] * M_2
        L_W2 = b[7] * M_6 + b[5] * M_4 + b[3] * M_2

        L_Z1 = b[12] * M_6 + b[10] * M_4 + b[8] * M_2
        L_Z2 = b[6] * M_6 + b[4] * M_4 + b[2] * M_2

        L_W = A_6 @ L_W1 + M_6 @ W_1 + L_W2
        L_U = A @ L_W + E @ W
        L_V = A_6 @ L_Z1 + M_6 @ Z_1 + L_Z2

    lu_decom = torch.lu(-U + V)
    exp_A = torch.lu_solve(U + V, *lu_decom)
    dexp_A = torch.lu_solve(L_U + L_V + (L_U - L_V) @ exp_A, *lu_decom)

    return exp_A, dexp_A


class expm(Function):
    """Computes the matrix exponential.
    """

    @staticmethod
    def forward(ctx, M):
        expm_M = _expm_scaling_squaring(M)
        ctx.save_for_backward(M)
        return expm_M

    @staticmethod
    def backward(ctx, grad_out):
        M = ctx.saved_tensors[0]
        dexpm = _expm_frechet_scaling_squaring(
            M, grad_out, adjoint=True)
        return dexpm



