// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(TreeLineMinDistance,
    ASCENDC_TPL_DATATYPE_DECL(DT_TREE_POINTS, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_BOOL_DECL(SCALAR, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_TREE_POINTS, C_DT_FLOAT, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(SCALAR, 0, 1),
    ),
);