
import copy

class LayerPlaceholder:
    def __init__(self,
        lid='0',
        input_ids=[''],
        read_count=0,
        input_shapes=[[]],
        output_shape=[],
        desc='the original layer converted from is ...'
    ):
        self._id = copy.deepcopy(lid)
        self._input_ids = copy.deepcopy(input_ids)
        self._read_count = copy.deepcopy(read_count)
        self._input_shapes = copy.deepcopy(input_shapes)
        self._output_shape = copy.deepcopy(output_shape)
        self._desc = copy.deepcopy(desc)

class LayerDemonstrate:
    def __init__(self,
        lid='0',
        input_ids=[''],
        read_count=0,
        input_shapes=[[]],
        output_shape=[],
        desc='the original layer converted from is ...'
    ):
        self._id = copy.deepcopy(lid)
        self._input_ids = copy.deepcopy(input_ids)
        self._read_count = copy.deepcopy(read_count)
        self._input_shapes = copy.deepcopy(input_shapes)
        self._output_shape = copy.deepcopy(output_shape)
        self._desc = copy.deepcopy(desc)
