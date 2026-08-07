# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses

import numpy as np

from census_bluedown import block
from absl.testing import absltest
from absl.testing import parameterized


class BlockTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           shape=(3, 4),
           num_asymmetric_features=1,
           asymmetric_features=(0,),
           symmetric_features=(1,),
           num_features=2,
           length=12,
           asymmetric_length=3,
           asymmetric_shape=(3, 1),
           uncompressed_symmetric_length=4,
           compressed_symmetric_length=2,
           compressed_length=6),
      dict(testcase_name="all_symmetric",
           shape=(3, 4),
           num_asymmetric_features=0,
           asymmetric_features=(),
           symmetric_features=(0, 1),
           num_features=2,
           length=12,
           asymmetric_length=1,
           asymmetric_shape=(1, 1),
           uncompressed_symmetric_length=12,
           compressed_symmetric_length=4,
           compressed_length=4),
      dict(testcase_name="all_asymmetric",
           shape=(3, 4),
           num_asymmetric_features=2,
           asymmetric_features=(0, 1),
           symmetric_features=(),
           num_features=2,
           length=12,
           asymmetric_length=12,
           asymmetric_shape=(3, 4),
           uncompressed_symmetric_length=1,
           compressed_symmetric_length=1,
           compressed_length=12),
      dict(testcase_name="multiple_of_each",
           shape=(3, 4, 5, 6),
           num_asymmetric_features=2,
           num_features=4,
           asymmetric_features=(0, 1),
           symmetric_features=(2, 3),
           length=360,
           asymmetric_length=12,
           asymmetric_shape=(3, 4, 1, 1),
           uncompressed_symmetric_length=30,
           compressed_symmetric_length=4,
           compressed_length=48),
      dict(testcase_name="default",
           shape=(8, 2, 2, 63),
           num_asymmetric_features=3,
           asymmetric_features=(0, 1, 2),
           symmetric_features=(3,),
           num_features=4,
           length=2016,
           asymmetric_length=32,
           asymmetric_shape=(8, 2, 2, 1),
           uncompressed_symmetric_length=63,
           compressed_symmetric_length=2,
           compressed_length=64),
  )
  def test_BlockShape_init(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      asymmetric_features: tuple[int, ...],
      symmetric_features: tuple[int, ...],
      num_features: int,
      length: int,
      asymmetric_length: int,
      asymmetric_shape: tuple[int, ...],
      uncompressed_symmetric_length: int,
      compressed_symmetric_length: int,
      compressed_length: int,
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    expected_dict = {
        "shape": shape,
        "num_asymmetric_features": num_asymmetric_features,
        "num_features": num_features,
        "asymmetric_features": asymmetric_features,
        "symmetric_features": symmetric_features,
        "length": length,
        "asymmetric_length": asymmetric_length,
        "asymmetric_shape": asymmetric_shape,
        "uncompressed_symmetric_length": uncompressed_symmetric_length,
        "compressed_symmetric_length": compressed_symmetric_length,
        "compressed_length": compressed_length,
        "asymmetric_queries": (),
        "asymmetric_query_dict": {}
    }
    self.assertDictEqual(dataclasses.asdict(block_shape), expected_dict)

  @parameterized.named_parameters(
      dict(testcase_name="first_feature",
           num_features=2,
           sliced_feature=0,
           partition=(3, 2),
           expected_shape=(2, 1)),
      dict(testcase_name="second_feature",
           num_features=2,
           sliced_feature=1,
           partition=(4, 6),
           expected_shape=(1, 2)),
      dict(testcase_name="default",
           num_features=4,
           sliced_feature=0,
           partition=(1, 4, 3),
           expected_shape=(3, 1, 1, 1)),
  )
  def test_AsymmetricQuery_init(
      self,
      num_features: int,
      sliced_feature: int,
      partition: tuple[int, ...],
      expected_shape: tuple[int, ...],
  ):
    query = block.AsymmetricQuery(num_features, sliced_feature, partition)
    self.assertEqual(query.shape, expected_shape)

  @parameterized.named_parameters(
      dict(testcase_name="symmetric_only_sliced",
           shape=(4,),
           num_asymmetric_features=0,
           query_shape=(4,),
           expected=[[1, 0],
                     [0, 1]]),
      dict(testcase_name="symmetric_only_aggregated",
           shape=(4,),
           num_asymmetric_features=0,
           query_shape=(1,),
           expected=[[0, 1]]),
      dict(testcase_name="symmetric_only_two_sliced",
           shape=(3, 4),
           num_asymmetric_features=0,
           query_shape=(3, 4),
           expected=[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
      dict(testcase_name="symmetric_only_first_sliced",
           shape=(3, 4),
           num_asymmetric_features=0,
           query_shape=(3, 1),
           expected=[[0, 1, 0, 0],
                     [0, 0, 0, 1]]),
      dict(testcase_name="symmetric_only_second_sliced",
           shape=(3, 4),
           num_asymmetric_features=0,
           query_shape=(1, 4),
           expected=[[0, 0, 1, 0],
                     [0, 0, 0, 1]]),
      dict(testcase_name="symmetric_only_two_aggregated",
           shape=(3, 4),
           num_asymmetric_features=0,
           query_shape=(1, 1),
           expected=[[0, 0, 0, 1]]),
      dict(testcase_name="total",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(1, 1),
           expected=[[0, 1, 0, 1, 0, 1]]),
      dict(testcase_name="slice_symmetric",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(1, 4),
           expected=[[1, 0, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 1]]),
      dict(testcase_name="slice_asymmetric",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(3, 1),
           expected=[[0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1]]),
      dict(testcase_name="detailed", shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(3, 4),
           expected=[[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]),
      dict(testcase_name="asymmetric_only_two_sliced",
           shape=(2, 3),
           num_asymmetric_features=2,
           query_shape=(2, 3),
           expected=[[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]),
      dict(testcase_name="asymmetric_only_first_sliced",
           shape=(2, 3),
           num_asymmetric_features=2,
           query_shape=(2, 1),
           expected=[[1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1]]),
      dict(testcase_name="asymmetric_only_second_sliced",
           shape=(2, 3),
           num_asymmetric_features=2,
           query_shape=(1, 3),
           expected=[[1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1]]),
      dict(testcase_name="asymmetric_only_two_aggregated",
           shape=(2, 3),
           num_asymmetric_features=2,
           query_shape=(1, 1),
           expected=[[1, 1, 1, 1, 1, 1]]),
  )
  def test_symmetric_query_design_rows(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      query_shape: tuple[int, ...],
      expected: list[list[int]],
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = block_shape.symmetric_query_design_rows(query_shape)
    self.assertAlmostEqual(computed.tolist(), expected)

  @parameterized.named_parameters(
      dict(testcase_name="one_asymmetric",
           shape=(4, 2, 10),
           num_asymmetric_features=1,
           sliced_feature=0,
           partition=(1, 3),
           expected=[[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]),
      dict(testcase_name="two_asymmetric",
           shape=(4, 2, 10),
           num_asymmetric_features=2,
           sliced_feature=0,
           partition=(1, 3),
           expected=[[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
           ),
      dict(testcase_name="second_sliced",
           shape=(2, 4, 10),
           num_asymmetric_features=2,
           sliced_feature=1,
           partition=(1, 3),
           expected=[[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1]])
  )
  def test_asymmetric_query_design_rows(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      sliced_feature: int,
      partition: tuple[int, ...],
      expected: list[list[int]],
  ):
    query = block.AsymmetricQuery(len(shape), sliced_feature, partition)
    block_shape = block.BlockShape(shape, num_asymmetric_features, (query,))
    computed = block_shape.query_design_rows(query.shape)
    self.assertAlmostEqual(computed.tolist(), expected)

  @parameterized.named_parameters(
      dict(testcase_name="symmetric_query",
           shape=(4,),
           num_asymmetric_features=1,
           sliced_feature=0,
           partition=(1, 3),
           query_shape=(4,),
           expected=[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
      dict(testcase_name="asymmetric_query",
           shape=(4,),
           num_asymmetric_features=1,
           sliced_feature=0,
           partition=(1, 3),
           query_shape=(2,),
           expected=[[1, 0, 0, 0],
                     [0, 1, 1, 1]]),
  )
  def test_query_design_rows(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      sliced_feature: int,
      partition: tuple[int, ...],
      query_shape: tuple[int, ...],
      expected: list[list[int]],
  ):
    query = block.AsymmetricQuery(len(shape), sliced_feature, partition)
    block_shape = block.BlockShape(shape, num_asymmetric_features, (query,))
    computed = block_shape.query_design_rows(query_shape)
    self.assertAlmostEqual(computed.tolist(), expected)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(3, 4),
           variance=1.0,
           expected=[[1.0, 1.0],
                     [1.0, 4.0]]),
      dict(testcase_name="variance",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(3, 4),
           variance=1.5,
           expected=[[1.5, 1.5],
                     [1.5, 6.0]]),
      dict(testcase_name="two_symmetric",
           shape=(3, 4, 5),
           num_asymmetric_features=1,
           query_shape=(3, 4, 5),
           variance=1.0,
           expected=[[1.0, 1.0, 1.0, 1.0],
                     [1.0, 5.0, 1.0, 5.0],
                     [1.0, 1.0, 4.0, 4.0],
                     [1.0, 5.0, 4.0, 20.0]]),
      dict(testcase_name="two_symmetric_variance_2",
           shape=(3, 4, 5),
           num_asymmetric_features=1,
           query_shape=(3, 4, 5),
           variance=2.0,
           expected=[[2.0, 2.0, 2.0, 2.0],
                     [2.0, 10.0, 2.0, 10.0],
                     [2.0, 2.0, 8.0, 8.0],
                     [2.0, 10.0, 8.0, 40.0]]),
      dict(testcase_name="sliced_symmetric_1",
           shape=(3, 4, 5),
           num_asymmetric_features=1,
           query_shape=(3, 1, 5),
           variance=2.0,
           expected=[[2.0, 2.0],
                     [2.0, 10.0]]),
      dict(testcase_name="sliced_symmetric_2",
           shape=(3, 4, 5),
           num_asymmetric_features=1,
           query_shape=(3, 4, 1),
           variance=2.0,
           expected=[[2.0, 2.0],
                     [2.0, 8.0]])
  )
  def test_covariance_block(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      query_shape: tuple[int, ...],
      variance: float,
      expected: list[list[float]],
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = block_shape.covariance_block(query_shape, variance)
    self.assertAlmostEqual(computed.tolist(), expected)

  def test_asymmetric_query_covariance_block(self):
    asymmetric_query = block.AsymmetricQuery(num_features=4,
                                             sliced_feature=0,
                                             partition=(1, 4, 3))
    block_shape = block.BlockShape(shape=(8, 2, 2, 63),
                                   num_asymmetric_features=3,
                                   asymmetric_queries=(asymmetric_query,))
    variance = 1.7
    computed = block_shape.covariance_block(asymmetric_query.shape, variance)
    expected = [[variance]]
    self.assertEqual(computed.tolist(), expected)

  @parameterized.named_parameters(
      dict(testcase_name="asymmetric",
           shape=(2,),
           num_asymmetric_features=1,
           query_shapes=((1,), (2,)),
           expected=[[1, 1],
                     [1, 0],
                     [0, 1]]),
      dict(testcase_name="symmetric",
           shape=(2,),
           num_asymmetric_features=0,
           query_shapes=((1,), (2,)),
           expected=[[0, 1],
                     [1, 0],
                     [0, 1]]),
  )
  def test_design_matrix(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      query_shapes: tuple[block.QueryShape, ...],
      expected: list[list[int]],
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = block_shape.design_matrix(query_shapes)
    self.assertEqual(computed.tolist(), expected)

  @parameterized.named_parameters(
      dict(testcase_name="asymmetric",
           shape=(2,),
           num_asymmetric_features=1,
           query_shapes=((1,), (2,)),
           query_variances=(1.0, 3.0),
           expected=[[1.0, 0.0, 0.0],
                     [0.0, 3.0, 0.0],
                     [0.0, 0.0, 3.0]]),
      dict(testcase_name="symmetric",
           shape=(2,),
           num_asymmetric_features=0,
           query_shapes=((1,), (2,)),
           query_variances=(1.0, 3.0),
           expected=[[1.0, 0.0, 0.0],
                     [0.0, 3.0, 3.0],
                     [0.0, 3.0, 6.0]]),
  )
  def test_covariance_matrix(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      query_shapes: tuple[block.QueryShape, ...],
      query_variances: tuple[float, ...],
      expected: list[list[float]],
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = block_shape.covariance_matrix(query_shapes, query_variances)
    self.assertAlmostEqual(computed.tolist(), expected)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(3, 1),
           expected=((0,), (1,), (2,))),
      dict(testcase_name="all_asymmetric",
           shape=(3, 2),
           num_asymmetric_features=2,
           query_shape=(3, 2),
           expected=((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),),
      dict(testcase_name="all_symmetric",
           shape=(3, 4),
           num_asymmetric_features=0,
           query_shape=(3, 1),
           expected=((),)),
      dict(testcase_name="total_query",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(1, 1),
           expected=((0,),)),
      dict(testcase_name="multiple_of_each",
           shape=(3, 2, 5, 6),
           num_asymmetric_features=2,
           query_shape=(3, 2, 1, 6),
           expected=((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))),
      dict(testcase_name="simplified_default",
           shape=(3, 2, 2, 63),
           num_asymmetric_features=3,
           query_shape=(3, 2, 2, 63),
           expected=((0, 0, 0,), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
                     (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 0, 1),
                     (2, 1, 0), (2, 1, 1))),
      dict(testcase_name="simplified_default_2",
           shape=(3, 2, 2, 63),
           num_asymmetric_features=3,
           query_shape=(1, 1, 1, 63),
           expected=((0, 0, 0,),)),
  )
  def test_asymmetric_feature_values(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      query_shape: tuple[int, ...],
      expected: tuple[tuple[int, ...], ...],
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = tuple(block_shape.asymmetric_feature_values(query_shape))
    self.assertTupleEqual(expected, computed)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(1, 4),
           expected=((4,), (1,))),
      dict(testcase_name="all_asymmetric",
           shape=(3, 4),
           num_asymmetric_features=2,
           query_shape=(1, 4),
           expected=((),)),
      dict(testcase_name="all_symmetric",
           shape=(3, 4),
           num_asymmetric_features=0,
           query_shape=(3, 4),
           expected=((3, 4), (3, 1), (1, 4), (1, 1))),
      dict(testcase_name="total_query",
           shape=(3, 4),
           num_asymmetric_features=1,
           query_shape=(1, 1),
           expected=((1,),)),
      dict(testcase_name="multiple_of_each",
           shape=(3, 4, 5, 6),
           num_asymmetric_features=2,
           query_shape=(3, 1, 5, 6),
           expected=((5, 6), (5, 1), (1, 6), (1, 1))),
      dict(testcase_name="default",
           shape=(8, 2, 2, 63),
           num_asymmetric_features=3,
           query_shape=(8, 2, 2, 63),
           expected=((63,), (1,))),
  )
  def test_symmetric_partial_rollups(
      self,
      shape: tuple[int, ...],
      num_asymmetric_features: int,
      query_shape: tuple[int, ...],
      expected: tuple[tuple[int, ...], ...],
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = tuple(block_shape.symmetric_partial_rollups(query_shape))
    self.assertTupleEqual(expected, computed)

  @parameterized.named_parameters(
      dict(
          testcase_name="asymmetric_only",
          shape=(3,),
          num_asymmetric_features=1,
          asymmetric_feature_values=(0,),
          symmetric_partial_rollup=(),
          query_value=np.array([5.0, 6.0, 7.0]),
          expected=np.array([5.0]),
          # Broadcasts to np.array([5.0, 5.0, 5.0])
      ),
      dict(
          testcase_name="symmetric_only",
          shape=(3,),
          num_asymmetric_features=0,
          asymmetric_feature_values=(),
          symmetric_partial_rollup=(3,),
          query_value=np.array([5.0, 6.0, 7.0]),
          expected=np.array([5.0, 6.0, 7.0]),
      ),
      dict(
          testcase_name="slice_symmetric",
          shape=(2, 3),
          num_asymmetric_features=1,
          asymmetric_feature_values=(0,),
          symmetric_partial_rollup=(3,),
          query_value=np.array([[4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]]),
          expected=np.array([[4.0, 5.0, 6.0]]),
          # Broadcasts to np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
      ),
      dict(
          testcase_name="aggregate_symmetric",
          shape=(2, 3),
          num_asymmetric_features=1,
          asymmetric_feature_values=(0,),
          symmetric_partial_rollup=(1,),
          query_value=np.array([[4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]]),
          expected=np.array([[15.0]]),
          # Broadcasts to np.array([[15.0, 15.0, 15.0], [15.0, 15.0, 15.0]])
      ),
      dict(
          testcase_name="multiple_features",
          shape=(2, 2, 2, 2),
          num_asymmetric_features=2,
          asymmetric_feature_values=(1, 0),
          symmetric_partial_rollup=(1, 2),
          query_value=np.array([[[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]]],
                                [[[9.0, 10.0], [11.0, 12.0]],
                                 [[13.0, 14.0], [15.0, 16.0]]]]),
          # Select the second value for the first feature and the first value
          # for the second feature, then aggregate over the third feature and
          # keep both values for the fourth feature. Since 9.0 + 11.0 = 20.0
          # and 10.0 + 12.0 = 22.0, the expected output is the following:
          expected=np.array([[[[20.0, 22.0]]]]),
          # Broadcasts to np.array([[[[20.0, 22.0], [20.0, 22.0]],
          #                          [[20.0, 22.0], [20.0, 22.0]]],
          #                         [[[20.0, 22.0], [20.0, 22.0]],
          #                          [[20.0, 22.0], [20.0, 22.0]]]]),
      ),
  )
  def test_query_aggregates(
      self,
      shape: block.QueryShape,
      num_asymmetric_features: int,
      asymmetric_feature_values: tuple[int, ...],
      symmetric_partial_rollup: tuple[int, ...],
      query_value: np.ndarray,
      expected: np.ndarray,
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = block_shape.query_aggregates(
        asymmetric_feature_values=asymmetric_feature_values,
        symmetric_partial_rollup=symmetric_partial_rollup,
        query_value=query_value)
    np.testing.assert_allclose(computed, expected)


if __name__ == "__main__":
  absltest.main()
