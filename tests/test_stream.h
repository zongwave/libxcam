/*
 * test_stream.h - test stream class
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_TEST_STREAM_H
#define XCAM_TEST_STREAM_H

#include <buffer_pool.h>
#include <image_file.h>
#if (!defined(ANDROID) && (HAVE_OPENCV))
#include "ocv/cv_utils.h"
#endif

#include <sys/stat.h>
#include <fcntl.h>

#define XCAM_TEST_STREAM_DEBUG 0
#define XCAM_TEST_STREAM_FOLDER "."

#define XCAM_TEST_MAX_STR_SIZE 256

#if (!defined(ANDROID) && (HAVE_OPENCV))
#define XCAM_TEST_OPENCV 1
#else
#define XCAM_TEST_OPENCV 0
#endif

#if XCAM_TEST_OPENCV
const static cv::Scalar color = cv::Scalar (0, 0, 255);
const static int fontFace = cv::FONT_HERSHEY_COMPLEX;
#endif

namespace XCam {

enum TestFileFormat {
    FileNone,
    FileNV12,
    FileYUV420,
    FileMP4,
    FileFIFO
};

#define PUSH_STREAM(Type, streams, file_name) \
    {                                                  \
        SmartPtr<Type> stream = new Type (file_name);  \
        XCAM_ASSERT (stream.ptr ());                   \
        streams.push_back (stream);                    \
    }

template <typename TType>
XCamReturn check_streams (const TType &streams)
{
    for (uint32_t i = 0; i < streams.size (); ++i) {
        if (!streams[i].ptr()) {
            XCAM_LOG_ERROR ("streams[%d] ptr is NULL", i);
            return XCAM_RETURN_ERROR_PARAM;
        }

        XCAM_FAIL_RETURN (
            ERROR, streams[i]->get_width () && streams[i]->get_height (), XCAM_RETURN_ERROR_PARAM,
            "streams[%d]: invalid parameters width:%d height:%d, please set buffer size first",
            i, streams[i]->get_width (), streams[i]->get_height ());
    }

    return XCAM_RETURN_NO_ERROR;
}

class Stream {
public:
    explicit Stream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~Stream ();

    void set_buf_size (uint32_t width, uint32_t height);
    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }
    const char *get_file_name () const {
        return _file_name;
    }
    void set_file (const SmartPtr<ImageFile> &file) {
        _file = file;
    }

    SmartPtr<VideoBuffer> &get_buf ();
    XCamReturn estimate_file_format ();

    XCamReturn open_reader (const char *option);
    XCamReturn open_writer (const char *option);
    XCamReturn close_file ();
    XCamReturn rewind ();

    XCamReturn read_buf ();
    XCamReturn write_buf (char *frame_str = NULL);
    virtual XCamReturn create_buf_pool (uint32_t reserve_count, uint32_t format = V4L2_PIX_FMT_NV12) = 0;

    XCamReturn open_fifo (int flag);
    XCamReturn close_fifo ();
    XCamReturn write_fifo (const SmartPtr<VideoBuffer> &buf);

#if XCAM_TEST_OPENCV
    void debug_write_image (char *img_name, char *frame_str = NULL, char *idx_str = NULL);
#endif

protected:
    void set_buf_pool (const SmartPtr<BufferPool> &pool) {
        _pool = pool;
    }

private:
#if XCAM_TEST_OPENCV
    XCamReturn cv_open_writer ();
    void cv_write_buf (char *frame_str = NULL);
#endif

private:
    XCAM_DEAD_COPY (Stream);

private:
    char                    *_file_name;
    uint32_t                 _width;
    uint32_t                 _height;

    SmartPtr<VideoBuffer>    _buf;
    SmartPtr<BufferPool>     _pool;

    SmartPtr<ImageFile>      _file;
    int                      _fifo;
#if XCAM_TEST_OPENCV
    cv::VideoWriter          _writer;
#endif
    TestFileFormat           _format;
};

Stream::Stream (const char *file_name, uint32_t width, uint32_t height)
    : _file_name (NULL)
    , _width (width)
    , _height (height)
    , _fifo (-1)
    , _format (FileNV12)
{
    if (file_name)
        _file_name = strndup (file_name, XCAM_TEST_MAX_STR_SIZE);
}

Stream::~Stream ()
{
    if (_file.ptr ()) {
        _file->close ();
        _file.release ();
    }

    if (_file_name) {
        xcam_free (_file_name);
        _file_name = NULL;
    }

    close_fifo ();
}

void
Stream::set_buf_size (uint32_t width, uint32_t height)
{
    _width = width;
    _height = height;
}

XCamReturn
Stream::open_reader (const char *option)
{
    XCAM_FAIL_RETURN (
        ERROR, (_format == FileNV12) || (_format == FileYUV420), XCAM_RETURN_ERROR_PARAM,
        "stream(%s) only support NV12 or YUV420 input format", _file_name);

    if (!_file.ptr ()) {
        SmartPtr<ImageFile> file = new ImageFile ();
        XCAM_ASSERT (file.ptr ());
        _file = file;
    }

    if (_file->open (_file_name, option) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("stream(%s) open failed", _file_name);
        return XCAM_RETURN_ERROR_FILE;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::open_writer (const char *option)
{
    XCAM_ASSERT (_format != FileNone);

    if (_format == FileNV12 || _format == FileYUV420) {
        if (!_file.ptr ()) {
            SmartPtr<ImageFile> file = new ImageFile ();
            XCAM_ASSERT (file.ptr ());
            _file = file;
        }

        if (_file->open (_file_name, option) != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("stream(%s) open failed", _file_name);
            return XCAM_RETURN_ERROR_FILE;
        }
    } else if (_format == FileMP4) {
#if XCAM_TEST_OPENCV
        XCamReturn ret = cv_open_writer ();
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "stream(%s) cv open writer failed", _file_name);
#else
        XCAM_LOG_ERROR ("stream(%s) unsupported MP4 format without opencv", _file_name);
        return XCAM_RETURN_ERROR_PARAM;
#endif
    } else if (_format == FileFIFO) {
        XCamReturn ret = open_fifo (0666);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "stream(%s) open FIFO failed", _file_name);
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", _file_name, (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Stream::close_file ()
{
    return _file->close ();
}

XCamReturn
Stream::rewind ()
{
    return _file->rewind ();
}

XCamReturn
Stream::read_buf ()
{
    XCAM_ASSERT (_pool.ptr ());

    _buf = _pool->get_buffer (_pool);
    XCAM_ASSERT (_buf.ptr ());

    return _file->read_buf (_buf);
}

XCamReturn
Stream::write_buf (char *frame_str) {
    if (_format == FileNV12 || _format == FileYUV420) {
        _file->write_buf (_buf);
    } else if (_format == FileMP4) {
#if XCAM_TEST_OPENCV
        cv_write_buf (frame_str);
#else
        XCAM_UNUSED (frame_str);
        XCAM_LOG_ERROR ("stream(%s) unsupported MP4 format without opencv", _file_name);
        return XCAM_RETURN_ERROR_PARAM;
#endif
    } else if (_format == FileFIFO) {
        write_fifo (_buf);
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %d", _file_name, (int)_format);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<VideoBuffer> &
Stream::get_buf ()
{
    if (!_buf.ptr () && _pool.ptr ()) {
        _buf = _pool->get_buffer (_pool);
        XCAM_ASSERT (_buf.ptr ());
    }

    return _buf;
}

XCamReturn
Stream::estimate_file_format ()
{
    XCAM_ASSERT (get_file_name ());

    char suffix[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    const char *ptr = strrchr (get_file_name (), '.');
    snprintf (suffix, XCAM_TEST_MAX_STR_SIZE, "%s", ptr + 1);

    if (!strcasecmp (suffix, "nv12")) {
        _format = FileNV12;
    } else if (!strcasecmp (suffix, "yuv")) {
        _format = FileYUV420;
    } else if (!strcasecmp (suffix, "mp4")) {
#if XCAM_TEST_OPENCV
        _format = FileMP4;
#else
        XCAM_LOG_ERROR ("stream(%s) unsupported MP4 format without opencv", _file_name);
        return XCAM_RETURN_ERROR_PARAM;
#endif
    } else if (!strcasecmp (suffix, "fifo")) {
        _format = FileFIFO;
    } else {
        XCAM_LOG_ERROR ("stream(%s) invalid file format: %s", _file_name, suffix);
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}

#if XCAM_TEST_OPENCV
XCamReturn
Stream::cv_open_writer ()
{
    XCAM_FAIL_RETURN (
        ERROR, _width && _height, XCAM_RETURN_ERROR_PARAM,
        "stream(%s) invalid size width:%d height:%d", _file_name, _width, _height);

    cv::Size frame_size = cv::Size (_width, _height);
    if (!_writer.open (_file_name, cv::VideoWriter::fourcc ('X', '2', '6', '4'), 30, frame_size)) {
        XCAM_LOG_ERROR ("stream(%s) open file failed", _file_name);
        return XCAM_RETURN_ERROR_FILE;
    }

    return XCAM_RETURN_NO_ERROR;
}

void
Stream::cv_write_buf (char *frame_str)
{
    cv::Mat mat;

#if XCAM_TEST_STREAM_DEBUG
    convert_to_mat (_buf, mat);
    cv::putText (mat, frame_str, cv::Point(20, 50), fontFace, 2.0, color, 2, 8, false);
#else
    XCAM_UNUSED (frame_str);
#endif

    if (_writer.isOpened ()) {
        if (mat.empty())
            convert_to_mat (_buf, mat);

        _writer.write (mat);
    }
}

void
Stream::debug_write_image (char *img_name, char *frame_str, char *idx_str)
{
    XCAM_ASSERT (img_name);

    cv::Mat mat;
    convert_to_mat (_buf, mat);

    if(frame_str)
        cv::putText (mat, frame_str, cv::Point(20, 50), fontFace, 2.0, color, 2, 8, false);
    if(idx_str)
        cv::putText (mat, idx_str, cv::Point(20, 110), fontFace, 2.0, color, 2, 8, false);

    cv::imwrite (img_name, mat);
}
#endif

XCamReturn
Stream::open_fifo (int flag)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (0 == mkfifo (_file_name, S_IFIFO | 0666)) {
        _fifo = open (_file_name, O_WRONLY);
    } else {
        if (EEXIST == errno) {
            XCAM_LOG_WARNING ("FIFO exist (%s) : %s", _file_name, strerror (errno));
            if (0 == unlink (_file_name)) {
                if (0 == mkfifo (_file_name, S_IFIFO | 0666)) {
                    _fifo = open (_file_name, O_WRONLY);
                } else {
                    ret = XCAM_RETURN_ERROR_FILE;
                    XCAM_LOG_ERROR ("create FIFO (%s) failed: %s", _file_name, strerror (errno));
                }
            } else {
                ret = XCAM_RETURN_ERROR_FILE;
                XCAM_LOG_ERROR ("delete FIFO (%s) failed: %s", _file_name, strerror (errno));
            }
        } else {
            ret = XCAM_RETURN_ERROR_FILE;
            XCAM_LOG_ERROR ("create FIFO (%s) failed: %s", _file_name, strerror (errno));
        }
    }

    if (-1 == _fifo) {
        ret = XCAM_RETURN_ERROR_FILE;
        XCAM_LOG_ERROR ("create FIFO (%s) failed: %s", _file_name, strerror (errno));
    }
    return ret;
}

XCamReturn
Stream::close_fifo ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (-1 != _fifo) {
        close (_fifo);
    }
    unlink (_file_name);

    return ret;
}

XCamReturn
Stream::write_fifo (const SmartPtr<VideoBuffer> &buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_fifo != -1);

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (write (_fifo, memory + info.offsets [index] + i * info.strides [index], line_bytes) != line_bytes) {
                XCAM_LOG_ERROR ("write FIFO failed, size doesn't match");
                ret = XCAM_RETURN_ERROR_FILE;
            }
        }
    }
    buf->unmap ();
    return ret;
}

}
#endif // XCAM_TEST_STREAM_H
