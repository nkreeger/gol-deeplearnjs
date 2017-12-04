/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Array2D, NDArray, NDArrayMathGPU, Graph, Session} from 'deeplearn';
import { Tensor } from 'deeplearn/dist/graph/graph';
import { Scalar } from 'deeplearn/dist/math/ndarray';

/** TODO(kreeger): Port this when pad operation is implemented. */
function padArray(array: NDArray): Array2D<'int32'> {
  const x1 = array.shape[0];
  const x2 = array.shape[1];
  const pad = 1;

  const oldValues = array.getValues();
  const shape = [x1 + pad * 2, x2 + pad * 2];
  const values = [];

  let z = 0;
  for (let i = 0; i < shape[0]; i++) {
    let rangeStart = -1;
    let rangeEnd = -1;
    if (i > 0 && i < shape[0] - 1) {
      rangeStart = i * shape[1] + 1;
      rangeEnd = i * shape[1] + x2;
    }
    for (let j = 0; j < shape[1]; j++) {
      const v = i * shape[0] + j;
      if (v >= rangeStart && v <= rangeEnd) {
        values[v] = oldValues[z++];
      } else {
        values[v] = 0;
      }
    }
  }
  return Array2D.new(shape as [number, number], values, 'int32');
}

function testPrint(array: NDArray, size: number) {
  let t = [];
  let v = array.getValues();
  for (let i = 0; i < v.length; i++) {
    t.push(v[i]);
    if (t.length == size) {
      console.log(t);
      t = [];
    }
  }
  console.log('');
}

async function gol() {
  const graph = new Graph();

  const math = new NDArrayMathGPU();
  const session = new Session(graph, math);

  const x: Tensor = graph.placeholder('x', []);
  const a: Tensor = graph.variable('a', Scalar.new(Math.random()));
  const b: Tensor = graph.variable('a', Scalar.new(Math.random()));
  const y: Tensor = graph.add(a, b);

  function generateGolExample(size: number) {
    // const world = Array2D.randUniform([size - 2, size - 2], 0, 2, 'int32');
    const world = Array2D.new([size - 2, size - 2],
      [[0,0,0],
      [1,1,1],
      [0,0,0]], 'int32');
    const worldPadded = padArray(world);

    let oneValues = [];
    for (let i = 0; i < world.size; i++) {
      oneValues.push(1);
    }
    const onesLike = Array2D.new([size - 2, size - 2], oneValues, 'int32');

    testPrint(worldPadded, 5);

    let x = math.add(
        math.slice2D(worldPadded, [0, 0], [3, 3]),
        math.slice2D(worldPadded, [0, 1], [3, 3]));
    x = math.add(x, math.slice2D(worldPadded, [0, 2], [3, 3]));
    x = math.add(x, math.slice2D(worldPadded, [1, 0], [3, 3]));
    x = math.add(x, math.slice2D(worldPadded, [1, 2], [3, 3]));
    x = math.add(x, math.slice2D(worldPadded, [2, 0], [3, 3]));
    x = math.add(x, math.slice2D(worldPadded, [2, 1], [3, 3]));
    x = math.add(x, math.slice2D(worldPadded, [2, 2], [3, 3]));

    console.log('------------------------');
    testPrint(x, 3);


    const y = x.getValues();
    for (let i = 0; i < y.length; i++) {
      let survives = (y[i] == 2 || y[i] == 3);
      let rebirths = y[i] == 3;

      // TODO: create ones-like
      let survivors = survives ? math.slice2D(worldPadded, [1, 1], [3, 3]) : NDArray.zerosLike(world);
      let worldNext = rebirths ? onesLike : survivors;
      let worldNextPadded = padArray(worldNext);
      testPrint(worldNextPadded, 3);
    }

    // console.log('worldPadded', worldPadded.getValues());
    // console.log('x', x.getValues());

    // TODO - cell survives
    // TODO - cell rebirths
    // TODO - world next
  }

  function trainModel(size: number) {
  }

  await math.scope(async (keep, track) => {
    // TODO - delete this.
    let result: NDArray = session.eval(y, [{tensor: x, data: track(Scalar.new(4))}]);
    console.log('result', result.shape);
    console.log('result.getValues()', result.getValues());

    generateGolExample(5);

    let losses = [];
    let steps = [];
    for (let i = 0; i < 10000; i++) {
      // TODO run the graph.
    }
  });
}

gol();
