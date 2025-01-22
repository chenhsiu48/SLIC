#!/bin/bash

for i in {1..24}; do
    name=`printf "kodim%02d.png" "$i"`
    wget -P kodak "https://r0k.us/graphics/kodak/kodak/$name"
done
