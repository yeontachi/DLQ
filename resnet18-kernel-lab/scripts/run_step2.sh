#!/usr/bin/env bash
set -e
BUILD=build/step2_conv1_bn_relu
MANI=exports/resnet18/fp32
$BUILD --manifest $MANI \
       --input   $MANI/fixtures/input.bin \
       --expect  $MANI/fixtures/expected.bin