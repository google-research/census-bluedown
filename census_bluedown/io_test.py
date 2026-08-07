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

from unittest import mock

import numpy as np
import pandas as pd

from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import io
from absl.testing import absltest


Estimate = estimate.Estimate


class IoTest(googletest.TestCase):

  @mock.patch.object(pd, 'read_parquet')
  @mock.patch.object(io.gfile, 'Open')
  @mock.patch.object(io.gfile, 'ListDir')
  def test_InputFormatIO_read(
      self,
      mock_list_dir,
      mock_open,
      mock_read_parquet):
    path_prefix = './data'
    subtree_folder = 'State=01'
    read_only = True
    path_suffix = '.parquet/DPQuery/'
    input_format_io = io.InputFormatIO(
        path_prefix, subtree_folder, read_only, path_suffix
    )

    mock_list_dir.return_value = ['file.parquet']
    demo_df = pd.DataFrame({
        constants.ID: ['01', '02'],
        constants.VALUE: [np.array([1, 2]), np.array([3, 4])],
        constants.VARIANCE: [
            np.array([[1, 0], [0, 1]]),
            np.array([[2, 1], [1, 2]]),
        ],
    })
    mock_file = mock.Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_read_parquet.return_value = demo_df

    df_read = input_format_io.read('State', filename=None)

    pd.testing.assert_frame_equal(df_read, demo_df)
    mock_list_dir.assert_called_once_with(
        './data/State.parquet/DPQuery/State=01'
    )
    mock_open.assert_called_once_with(
        './data/State.parquet/DPQuery/State=01/file.parquet', 'rb'
    )
    mock_read_parquet.assert_called_once_with(mock_file,
                                              engine='pyarrow',
                                              use_threads=False)

  @mock.patch.object(io.gfile, 'MakeDirs')
  def test_InputFormatIO_write(self, mock_make_dirs):
    path_prefix = './data'
    subtree_folder = 'State=01'
    read_only = False
    path_suffix = '.parquet/DPQuery/'
    input_format_io = io.InputFormatIO(
        path_prefix, subtree_folder, read_only, path_suffix
    )

    mock_df = mock.Mock()
    mock_df.to_parquet.return_value = None

    input_format_io.write('State', 'file.parquet', mock_df)

    mock_make_dirs.assert_called_once_with(
        './data/State.parquet/DPQuery/State=01'
    )
    mock_df.to_parquet.assert_called_once_with(
        './data/State.parquet/DPQuery/State=01/file.parquet'
    )

  @mock.patch.object(io.gfile, 'ListDir')
  def test_InputFormatIO_exists(self, mock_list_dir):
    path_prefix = './data'
    subtree_folder = 'State=01'
    read_only = True
    path_suffix = '.parquet/DPQuery/'
    input_format_io = io.InputFormatIO(
        path_prefix, subtree_folder, read_only, path_suffix
    )

    mock_list_dir.return_value = []
    self.assertFalse(input_format_io.exists('State'))

    mock_list_dir.return_value = ['file1.parquet']
    self.assertTrue(input_format_io.exists('State'))

    mock_list_dir.return_value = ['file1.parquet', 'file2.parquet']
    self.assertTrue(input_format_io.exists('State'))

  @mock.patch.object(pd, 'read_parquet')
  @mock.patch.object(pd, 'read_pickle')
  def test_ProcessingFormatIO_read(self, mock_read_pickle, mock_read_parquet):
    path_prefix = './data'
    subtree_folder = 'State=01'
    read_only = True
    split_estimates = False
    pickle_compression = True
    processing_format_io = io.ProcessingFormatIO(
        path_prefix=path_prefix,
        subtree_folder=subtree_folder,
        read_only=read_only,
        split_estimates=split_estimates,
        pickle_compression=pickle_compression
    )

    demo_df_unpacked = pd.DataFrame({
        constants.ID: ['01', '02'],
        constants.VALUE: [np.array([1, 2]), np.array([3, 4])],
        constants.VARIANCE: [
            np.array([[1, 0, 0, 1]]),
            np.array([[2, 1, 1, 2]]),
        ],
    })
    demo_df_packed = io.pack_estimates(demo_df_unpacked.copy())
    mock_read_pickle.return_value = demo_df_packed

    df_read_packed = processing_format_io.read('State', 'estimate.pickle')
    pd.testing.assert_frame_equal(df_read_packed, demo_df_packed)
    mock_read_pickle.assert_called_once_with(
        './data/State=01/State/estimate.pickle'
    )

    split_estimates = True
    pickle_compression = False
    processing_format_io = io.ProcessingFormatIO(
        path_prefix=path_prefix,
        subtree_folder=subtree_folder,
        read_only=read_only,
        split_estimates=split_estimates,
        pickle_compression=pickle_compression
    )
    mock_read_parquet.return_value = demo_df_unpacked

    df_read_unpacked = processing_format_io.read('State', 'estimate.parquet')
    self.assertListEqual(
        list(df_read_unpacked[constants.ID]),
        list(demo_df_packed[constants.ID])
    )
    np.testing.assert_allclose(
        [x.val for x in df_read_unpacked[constants.ESTIMATE]],
        [y.val for y in demo_df_packed[constants.ESTIMATE]]
    )
    np.testing.assert_allclose(
        [x.cov for x in df_read_unpacked[constants.ESTIMATE]],
        [y.cov for y in demo_df_packed[constants.ESTIMATE]]
    )
    mock_read_parquet.assert_called_once_with(
        './data/State=01/State/estimate.parquet',
        engine='pyarrow',
        use_threads=False
    )

  @mock.patch.object(io, 'expand_estimates')
  @mock.patch.object(io.gfile, 'MakeDirs')
  def test_ProcessingFormatIO_write(
      self,
      mock_make_dirs,
      mock_unpack_estimates):
    path_prefix = './data'
    subtree_folder = 'State=01'
    read_only = False
    processing_format_io_packed = io.ProcessingFormatIO(
        path_prefix, subtree_folder, read_only,
        split_estimates=False,
        pickle_compression=True
    )
    processing_format_io_unpacked = io.ProcessingFormatIO(
        path_prefix, subtree_folder, read_only, split_estimates=True
    )

    mock_df = mock.Mock()
    mock_df.to_pickle.return_value = None
    mock_unpacked_df = mock.Mock()
    mock_unpacked_df.to_parquet.return_value = None
    mock_unpack_estimates.return_value = mock_unpacked_df

    processing_format_io_packed.write('State', 'file.pickle', mock_df)
    processing_format_io_unpacked.write('State', 'file.parquet', mock_df)

    self.assertEqual(mock_make_dirs.call_count, 2)
    mock_make_dirs.assert_has_calls(
        [mock.call('./data/State=01/State'),
         mock.call('./data/State=01/State')])
    mock_df.to_pickle.assert_called_once_with(
        './data/State=01/State/file.pickle'
    )
    mock_unpacked_df.to_parquet.assert_called_once_with(
        './data/State=01/State/file.parquet'
    )

  @mock.patch.object(io.gfile, 'Exists')
  def test_ProcessingFormatIO_exists(self, mock_exists):
    path_prefix = './data'
    subtree_folder = 'State=01'
    read_only = True
    split_estimates = False
    processing_format_io = io.ProcessingFormatIO(
        path_prefix, subtree_folder, read_only, split_estimates
    )
    directory_files = []

    def mock_exists_side_effect(filename):
      return filename in directory_files

    mock_exists.side_effect = mock_exists_side_effect
    self.assertFalse(processing_format_io.exists('State', 'file1.parquet'))

    directory_files = ['./data/State=01/State/file1.parquet']
    self.assertTrue(processing_format_io.exists('State', 'file1.parquet'))
    self.assertFalse(processing_format_io.exists('State', 'file2.parquet'))

    directory_files = ['./data/State=01/State/file1.parquet',
                       './data/State=01/State/file2.parquet']
    self.assertTrue(processing_format_io.exists('State', 'file1.parquet'))
    self.assertTrue(processing_format_io.exists('State', 'file2.parquet'))
    self.assertFalse(processing_format_io.exists('State', 'file3.parquet'))

  def test_pack_estimates(self):
    id1 = '01'
    id2 = '02'
    val1 = np.array([1, 2])
    val2 = np.array([3, 4])
    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[2, 1], [1, 2]])
    flat_cov1 = np.array([1, 0, 0, 1])
    flat_cov2 = np.array([2, 1, 1, 2])
    df = pd.DataFrame({
        constants.ID: [id1, id2],
        constants.VALUE: [val1, val2],
        constants.VARIANCE: [flat_cov1, flat_cov2],
    })
    df_packed = io.pack_estimates(df)
    expected_columns = (constants.ID, constants.ESTIMATE)
    self.assertEqual(tuple(df_packed.columns), expected_columns)
    pd.testing.assert_series_equal(
        df_packed[constants.ID],
        pd.Series([id1, id2], name=constants.ID)
    )
    np.testing.assert_allclose(
        [x.val for x in df_packed[constants.ESTIMATE]],
        [y.val for y in [Estimate(val1, cov1), Estimate(val2, cov2)]])
    np.testing.assert_allclose(
        [x.cov for x in df_packed[constants.ESTIMATE]],
        [y.cov for y in [Estimate(val1, cov1), Estimate(val2, cov2)]])

  def test_split_estimates(self):
    id1 = '01'
    id2 = '02'
    val1 = np.array([1, 2])
    val2 = np.array([3, 4])
    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[2, 1], [1, 2]])
    df = pd.DataFrame({
        constants.ID: [id1, id2],
        constants.ESTIMATE: [Estimate(val1, cov1), Estimate(val2, cov2)],
    })
    df_unpacked = io.expand_estimates(df)
    expected_columns = (constants.ID, constants.VALUE, constants.VARIANCE)
    self.assertEqual(tuple(df_unpacked.columns), expected_columns)
    pd.testing.assert_series_equal(
        df_unpacked[constants.ID],
        pd.Series([id1, id2], name=constants.ID)
    )
    pd.testing.assert_series_equal(
        df_unpacked[constants.VALUE],
        pd.Series([val1, val2], name=constants.VALUE)
    )
    pd.testing.assert_series_equal(
        df_unpacked[constants.VARIANCE],
        pd.Series([cov1.flatten(), cov2.flatten()], name=constants.VARIANCE)
    )


if __name__ == '__main__':
  absltest.main()
