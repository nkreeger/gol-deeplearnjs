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

import {Array2D, Graph, NDArray, NDArrayMathGPU, Session} from 'deeplearn';
import {Tensor} from 'deeplearn/dist/graph/graph';
import {Scalar} from 'deeplearn/dist/math/ndarray';

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

  function countNeighbors(size: number, worldPadded: Array2D): Array2D {
    let neighborCount = math.add(
        math.slice2D(worldPadded, [0, 0], [size - 2, size - 2]),
        math.slice2D(worldPadded, [0, 1], [size - 2, size - 2]));
    neighborCount = math.add(
        neighborCount, math.slice2D(worldPadded, [0, 2], [size - 2, size - 2]));
    neighborCount = math.add(
        neighborCount, math.slice2D(worldPadded, [1, 0], [size - 2, size - 2]));
    neighborCount = math.add(
        neighborCount, math.slice2D(worldPadded, [1, 2], [size - 2, size - 2]));
    neighborCount = math.add(
        neighborCount, math.slice2D(worldPadded, [2, 0], [size - 2, size - 2]));
    neighborCount = math.add(
        neighborCount, math.slice2D(worldPadded, [2, 1], [size - 2, size - 2]));
    neighborCount = math.add(
        neighborCount, math.slice2D(worldPadded, [2, 2], [size - 2, size - 2]));
    return neighborCount as Array2D;
  }

  function cellSurvives(size: number, numNeigbors: Array2D): Array2D {
    const survives = Array2D.zeros([size - 2, size - 2], 'bool');
    //
    // TODO(kreeger): write me.
    //
    return survives;
  }

  function cellRebirths(size: number, numNeigbors: Array2D): Array2D {
    const rebirths = Array2D.zeros([size - 2, size - 2], 'bool');
    //
    // TODO(kreeger): write me.
    //
    return rebirths;
  }

  function generateGolExample(size: number) {
    // const world = Array2D.randUniform([size - 2, size - 2], 0, 2, 'int32');
    const world = Array2D.new(
        [size - 2, size - 2], [[0, 0, 0], [1, 1, 1], [0, 0, 0]], 'int32');
    const worldPadded = padArray(world);
    console.log('world padded ------------------------');
    testPrint(worldPadded, size);

    let numNeighbors = countNeighbors(size, worldPadded);
    console.log('num neighbors ------------------------');
    testPrint(numNeighbors, size - 2);

    // Cell survives
    console.log('cell survives ------------------------');
    const survives = cellSurvives(size, numNeighbors);
    testPrint(survives, size - 2);

    // Cell rebirths
    console.log('cell rebirths ------------------------');
    const rebirths = cellRebirths(size, numNeighbors);
    testPrint(rebirths, size - 2);

    // TODO - world next
  }

  function trainModel(size: number) {}

  await math.scope(async (keep, track) => {
    // TODO - delete this.
    let result: NDArray =
        session.eval(y, [{tensor: x, data: track(Scalar.new(4))}]);
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
