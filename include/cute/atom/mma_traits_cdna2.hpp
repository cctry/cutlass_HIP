#pragma once

#include <cute/arch/mma_cdna2.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{
    template <>
    struct MMA_Traits<CDNA2_32x32x2_F32F32F32F32_TN>
    {
    using ValTypeD = float;
    using ValTypeA = float;
    using ValTypeB = float;
    using ValTypeC = float;

    using Shape_MNK = Shape<_32,_32,_2>;
    using ThrID   = Layout<_64>;
    using ALayout = Layout<Shape<_64, _1>,
                            Stride<_1, _0>>;
    using BLayout = Layout<Shape<_64, _1>,
                            Stride<_1, _0>>;
    using CLayout = Layout<Shape<Shape<_32, _2>, Shape<_4, _4>>,
                            Stride<Stride<_32, _4>, Stride<_1, _8>>>;
    };

}