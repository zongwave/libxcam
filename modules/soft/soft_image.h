/*
 * soft_image.h - soft image class
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_SOFT_IMAGE_H
#define XCAM_SOFT_IMAGE_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <vec_mat.h>
#include <file_handle.h>

#if ENABLE_AVX512
#include <immintrin.h>
#endif

#if ENABLE_AVX512
#define XCAM_SOFT_WORKUNIT_PIXELS 16
#define DUMP_AVX_INTERPOLATION 0
#else
#define XCAM_SOFT_WORKUNIT_PIXELS 8
#endif

namespace XCam {

typedef uint8_t Uchar;
typedef int8_t Char;
typedef Vector2<uint8_t> Uchar2;
typedef Vector2<int8_t> Char2;
typedef Vector2<float> Float2;
typedef Vector2<int> Int2;

enum BorderType {
    BorderTypeNearest,
    BorderTypeConst,
    BorderTypeRewind,
};

template <typename T>
class SoftImage
{
public:
    typedef T Type;
private:
    uint8_t    *_buf_ptr;
    uint32_t    _width;
    uint32_t    _height;
    uint32_t    _pitch;

    SmartPtr<VideoBuffer> _bind;

public:
    explicit SoftImage (const SmartPtr<VideoBuffer> &buf, const uint32_t plane);
    explicit SoftImage (
        const uint32_t width, const uint32_t height,
        uint32_t aligned_width = 0);
    explicit SoftImage (
        const SmartPtr<VideoBuffer> &buf,
        const uint32_t width, const uint32_t height, const uint32_t pictch, const uint32_t offset = 0);

    ~SoftImage () {
        if (!_bind.ptr ()) {
            xcam_free (_buf_ptr);
        }
    }

    uint32_t pixel_size () const {
        return sizeof (T);
    }

    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }
    uint32_t get_pitch () const {
        return _pitch;
    }
    bool is_valid () const {
        return (_buf_ptr && _width && _height);
    }

    const SmartPtr<VideoBuffer> &get_bind_buf () const {
        return _bind;
    }
    T *get_buf_ptr (int32_t x, int32_t y) {
        return (T *)(_buf_ptr + y * _pitch) + x;
    }
    const T *get_buf_ptr (int32_t x, int32_t y) const {
        return (const T *)(_buf_ptr + y * _pitch) + x;
    }

    inline T read_data_no_check (int32_t x, int32_t y) const {
        const T *t_ptr = (const T *)(_buf_ptr + y * _pitch);
        return t_ptr[x];
    }

    inline T read_data (int32_t x, int32_t y) const {
        border_check (x, y);
        return read_data_no_check (x, y);
    }

    template<typename O>
    inline O read_interpolate_data (float x, float y) const;

    template<typename O, uint32_t N>
    inline void read_interpolate_array (Float2 *pos, O *array) const;

#if ENABLE_AVX512
    inline void read_interpolate_array (Float2 *pos, Float2 *array) const;
    inline void read_interpolate_array (Float2 *pos, Uchar *array) const;
    inline void read_interpolate_array (Float2 *pos, Uchar2 *array) const;
#endif

    template<uint32_t N>
    inline void read_array_no_check (const int32_t x, const int32_t y, T *array) const {
        XCAM_ASSERT (N <= 8);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch)) + x;
        memcpy (array, t_ptr, sizeof (T) * N);
    }

    template<typename O, uint32_t N>
    inline void read_array_no_check (const int32_t x, const int32_t y, O *array) const {
        XCAM_ASSERT (N <= 8);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch)) + x;
        for (uint32_t i = 0; i < N; ++i) {
            array[i] = t_ptr[i];
        }
    }

    template<uint32_t N>
    inline void read_array (int32_t x, int32_t y, T *array) const {
        XCAM_ASSERT (N <= 8);
        border_check_y (y);
        if (x + N < _width) {
            read_array_no_check<N> (x, y, array);
        } else {
            const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
            for (uint32_t i = 0; i < N; ++i, ++x) {
                border_check_x (x);
                array[i] = t_ptr[x];
            }
        }
    }

    template<typename O, uint32_t N>
    inline void read_array (int32_t x, int32_t y, O *array) const {
        XCAM_ASSERT (N <= 8);
        border_check_y (y);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
        for (uint32_t i = 0; i < N; ++i, ++x) {
            border_check_x (x);
            array[i] = t_ptr[x];
        }
    }

    inline void write_data (int32_t x, int32_t y, const T &v) {
        if (x < 0 || x >= (int32_t)_width)
            return;
        if (y < 0 || y >= (int32_t)_height)
            return;
        write_data_no_check (x, y, v);
    }

    inline void write_data_no_check (int32_t x, int32_t y, const T &v) {
        T *t_ptr = (T *)(_buf_ptr + y * _pitch);
        t_ptr[x] = v;
    }

    template<uint32_t N>
    inline void write_array_no_check (int32_t x, int32_t y, const T *array) {
        T *t_ptr = (T *)(_buf_ptr + y * _pitch);
        memcpy (t_ptr + x, array, sizeof (T) * N);
    }

    template<uint32_t N>
    inline void write_array (int32_t x, int32_t y, const T *array) {
        if (y < 0 || y >= (int32_t)_height)
            return;

        if (x >= 0 && x + N <= _width) {
            write_array_no_check<N> (x, y, array);
        } else {
            T *t_ptr = ((T *)(_buf_ptr + y * _pitch));
            for (uint32_t i = 0; i < N; ++i, ++x) {
                if (x < 0 || x >= (int32_t)_width) continue;
                t_ptr[x] = array[i];
            }
        }
    }

private:
    inline void border_check_x (int32_t &x) const {
        if (x < 0) x = 0;
        else if (x >= (int32_t)_width) x = (int32_t)(_width - 1);
    }

    inline void border_check_y (int32_t &y) const {
        if (y < 0) y = 0;
        else if (y >= (int32_t)_height) y = (int32_t)(_height - 1);
    }

    inline void border_check (int32_t &x, int32_t &y) const {
        border_check_x (x);
        border_check_y (y);
    }
};


template <typename T>
SoftImage<T>::SoftImage (const SmartPtr<VideoBuffer> &buf, const uint32_t plane)
    : _buf_ptr (NULL)
    , _width (0), _height (0), _pitch (0)
{
    XCAM_ASSERT (buf.ptr ());
    const VideoBufferInfo &info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    if (!info.get_planar_info(planar, plane)) {
        XCAM_LOG_ERROR (
            "videobuf to soft image failed. buf format:%s, plane:%d", xcam_fourcc_to_string (info.format), plane);
        return;
    }
    _buf_ptr = buf->map () + info.offsets[plane];
    XCAM_ASSERT (_buf_ptr);
    _pitch = info.strides[plane];
    _height = planar.height;
    _width = planar.pixel_bytes * planar.width / sizeof (T);
    XCAM_ASSERT (_width * sizeof(T) == planar.pixel_bytes * planar.width);
    _bind = buf;
}

template <typename T>
SoftImage<T>::SoftImage (
    const uint32_t width, const uint32_t height, uint32_t aligned_width)
    : _buf_ptr (NULL)
    , _width (0), _height (0), _pitch (0)
{
    if (!aligned_width)
        aligned_width = width;

    XCAM_ASSERT (aligned_width >= width);
    XCAM_ASSERT (width > 0 && height > 0);
    _pitch = aligned_width * sizeof (T);
    _buf_ptr = (uint8_t *)xcam_malloc (_pitch * height);
    XCAM_ASSERT (_buf_ptr);
    _width = width;
    _height = height;
}

template <typename T>
SoftImage<T>::SoftImage (
    const SmartPtr<VideoBuffer> &buf,
    const uint32_t width, const uint32_t height, const uint32_t pictch, const uint32_t offset)
    : _buf_ptr (NULL)
    , _width (width), _height (height)
    , _pitch (pictch)
    , _bind (buf)
{
    XCAM_ASSERT (buf.ptr ());
    XCAM_ASSERT (buf->map ());
    _buf_ptr = buf->map () + offset;
}

template <typename T>
inline Uchar convert_to_uchar (const T& v) {
    if (v < 0.0f) return 0;
    else if (v > 255.0f) return 255;
    return (Uchar)(v + 0.5f);
}

template <typename T, uint32_t N>
inline void convert_to_uchar_N (const T *in, Uchar *out) {
    for (uint32_t i = 0; i < N; ++i) {
        out[i] = convert_to_uchar<T> (in[i]);
    }
}

template <typename Vec2>
inline Uchar2 convert_to_uchar2 (const Vec2& v) {
    return Uchar2 (convert_to_uchar(v.x), convert_to_uchar(v.y));
}

template <typename Vec2, uint32_t N>
inline void convert_to_uchar2_N (const Vec2 *in, Uchar2 *out) {
    for (uint32_t i = 0; i < N; ++i) {
        out[i].x = convert_to_uchar (in[i].x);
        out[i].y = convert_to_uchar (in[i].y);
    }
}

typedef SoftImage<Uchar> UcharImage;
typedef SoftImage<Uchar2> Uchar2Image;
typedef SoftImage<float> FloatImage;
typedef SoftImage<Float2> Float2Image;

template <class SoftImageT>
class SoftImageFile
    : public FileHandle
{
public:
    SoftImageFile () {}
    explicit SoftImageFile (const char *name, const char *option)
        : FileHandle (name, option)
    {}

    inline XCamReturn read_buf (const SmartPtr<SoftImageT> &buf);
    inline XCamReturn write_buf (const SmartPtr<SoftImageT> &buf);
};

template <class SoftImageT>
inline XCamReturn
SoftImageFile<SoftImageT>::read_buf (const SmartPtr<SoftImageT> &buf)
{
    XCAM_FAIL_RETURN (
        WARNING, is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) read buf failed, file is not open", XCAM_STR (get_file_name ()));

    XCAM_FAIL_RETURN (
        WARNING, buf->is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) read buf failed, buf is not valid", XCAM_STR (get_file_name ()));

    XCAM_ASSERT (is_valid ());
    uint32_t height = buf->get_height ();
    uint32_t line_bytes = buf->get_width () * buf->pixel_size ();

    for (uint32_t index = 0; index < height; index++) {
        uint8_t *line_ptr = buf->get_buf_ptr (0, index);
        XCAM_FAIL_RETURN (
            WARNING, fread (line_ptr, 1, line_bytes, _fp) == line_bytes, XCAM_RETURN_ERROR_FILE,
            "soft image file(%s) read buf failed, image_line:%d", XCAM_STR (get_file_name ()), index);
    }
    return XCAM_RETURN_NO_ERROR;
}

template <class SoftImageT>
inline XCamReturn
SoftImageFile<SoftImageT>::write_buf (const SmartPtr<SoftImageT> &buf)
{
    XCAM_FAIL_RETURN (
        WARNING, is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) write buf failed, file is not open", XCAM_STR (get_file_name ()));

    XCAM_FAIL_RETURN (
        WARNING, buf->is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) write buf failed, buf is not valid", XCAM_STR (get_file_name ()));

    XCAM_ASSERT (is_valid ());
    uint32_t height = buf->get_height ();
    uint32_t line_bytes = buf->get_width () * buf->pixel_size ();

    for (uint32_t index = 0; index < height; index++) {
        uint8_t *line_ptr = buf->get_buf_ptr (0, index);
        XCAM_FAIL_RETURN (
            WARNING, fwrite (line_ptr, 1, line_bytes, _fp) == line_bytes, XCAM_RETURN_ERROR_FILE,
            "soft image file(%s) write buf failed, image_line:%d", XCAM_STR (get_file_name ()), index);
    }
    return XCAM_RETURN_NO_ERROR;
}

template <typename T> template <typename O>
O
SoftImage<T>::read_interpolate_data (float x, float y) const
{
    int32_t x0 = (int32_t)(x), y0 = (int32_t)(y);
    float a = x - x0, b = y - y0;
    O l0[2], l1[2];
    read_array<O, 2> (x0, y0, l0);
    read_array<O, 2> (x0, y0 + 1, l1);

    return l1[1] * (a * b) + l0[0] * ((1 - a) * (1 - b)) +
           l1[0] * ((1 - a) * b) + l0[1] * (a * (1 - b));
}

#if !ENABLE_AVX512
template <typename T> template<typename O, uint32_t N>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, O *array) const
{
    for (uint32_t i = 0; i < N; ++i) {
        array[i] = read_interpolate_data<O> (pos[i].x, pos[i].y);
    }
}
#endif

#if ENABLE_AVX512

template <typename T> template<typename O, uint32_t N>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, O *array) const
{
    Float2 weight[16];
    O tl[16];
    O tr[16];
    O bl[16];
    O br[16];
    for (uint32_t i = 0; i < N; ++i) {
        int32_t x0 = (int32_t)(pos[i].x);
        int32_t y0 = (int32_t)(pos[i].y);
        float a = pos[i].x - x0;
        float b = pos[i].y - y0;
        O l0[2], l1[2];

        weight[i].x = a;
        weight[i].y = b;

        int32_t x = x0;
        int32_t y = y0;

        border_check_y (y);
        const T *t_ptr0 = ((const T *)(_buf_ptr + y * _pitch));

        border_check_x (x);
        l0[0] = t_ptr0[x++];
        border_check_x (x);
        l0[1] = t_ptr0[x];

        x = x0, y = y0 + 1;
        border_check_y (y);
        const T *t_ptr1 = ((const T *)(_buf_ptr + y * _pitch));

        border_check_x (x);
        l1[0] = t_ptr1[x++];
        border_check_x (x);
        l1[1] = t_ptr1[x];

        tl[i] = l0[0];
        tr[i] = l0[1];
        bl[i] = l1[0];
        br[i] = l1[1];

        array[i] = l1[1] * (a * b) + l0[0] * ((1 - a) * (1 - b)) +
                   l1[0] * ((1 - a) * b) + l0[1] * (a * (1 - b));
    }

#if 0
    XCAM_LOG_ERROR ("\n @ZONGWAVE AVX-CPU interpolate coordinate \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_cpu_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tr_cpu_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n bl_cpu_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n br_cpu_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f]; \n interp_cpu_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n @ZONGWAVE AVX-CPU \n\n",
                    pos[0].x, pos[0].y, pos[1].x, pos[1].y, pos[2].x, pos[2].y, pos[3].x, pos[3].y,
                    pos[4].x, pos[4].y, pos[5].x, pos[5].y, pos[6].x, pos[6].y, pos[7].x, pos[7].y,
                    tl[0].x, tl[0].y, tl[1].x, tl[1].y,
                    tl[2].x, tl[2].y, tl[3].x, tl[3].y,
                    tl[4].x, tl[4].y, tl[5].x, tl[5].y,
                    tl[6].x, tl[6].y, tl[7].x, tl[7].y,
                    tr[0].x, tr[0].y, tr[1].x, tr[1].y,
                    tr[2].x, tr[2].y, tr[3].x, tr[3].y,
                    tr[4].x, tr[4].y, tr[5].x, tr[5].y,
                    tr[6].x, tr[6].y, tr[7].x, tr[7].y,
                    bl[0].x, bl[0].y, bl[1].x, bl[1].y,
                    bl[2].x, bl[2].y, bl[3].x, bl[3].y,
                    bl[4].x, bl[4].y, bl[5].x, bl[5].y,
                    bl[6].x, bl[6].y, bl[7].x, bl[7].y,
                    br[0].x, br[0].y, br[1].x, br[1].y,
                    br[2].x, br[2].y, br[3].x, br[3].y,
                    br[4].x, br[4].y, br[5].x, br[5].y,
                    br[6].x, br[6].y, br[7].x, br[7].y,
                    weight[0].x, weight[1].x, weight[2].x, weight[3].x,
                    weight[4].x, weight[5].x, weight[6].x, weight[7].x,
                    weight[0].y, weight[1].y, weight[2].y, weight[3].y,
                    weight[4].y, weight[5].y, weight[6].y, weight[7].y,
                    array[0].x, array[0].y, array[1].x, array[1].y,
                    array[2].x, array[2].y, array[3].x, array[3].y,
                    array[4].x, array[4].y, array[5].x, array[5].y,
                    array[6].x, array[6].y, array[7].x, array[7].y);
#endif
#if 0  // luma Y
    XCAM_LOG_ERROR ("\n @ZONGWAVE AVX-CPU interpolate luma \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_avx_pixel=[%f %f %f %f %f %f %f %f]; \n tr_avx_pixel=[%f %f %f %f %f %f %f %f]; \n bl_avx_pixel=[%f %f %f %f %f %f %f %f]; \n br_avx_pixel=[%f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f]; \n interp_avx_y_pixel=[%f %f %f %f %f %f %f %f]; \n \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_avx_pixel=[%f %f %f %f %f %f %f %f]; \n tr_avx_pixel=[%f %f %f %f %f %f %f %f]; \n bl_avx_pixel=[%f %f %f %f %f %f %f %f]; \n br_avx_pixel=[%f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f]; \n interp_avx_y_pixel=[%f %f %f %f %f %f %f %f]; \n @ZONGWAVE AVX-CPU \n\n",
                    pos[0].x, pos[0].y, pos[1].x, pos[1].y, pos[2].x, pos[2].y, pos[3].x, pos[3].y,
                    pos[4].x, pos[4].y, pos[5].x, pos[5].y, pos[6].x, pos[6].y, pos[7].x, pos[7].y,
                    tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7],
                    tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6], tr[7],
                    bl[0], bl[1], bl[2], bl[3], bl[4], bl[5], bl[6], bl[7],
                    br[0], br[1], br[2], br[3], br[4], br[5], br[6], br[7],
                    weight[0].x, weight[1].x, weight[2].x, weight[3].x,
                    weight[4].x, weight[5].x, weight[6].x, weight[7].x,
                    weight[0].y, weight[1].y, weight[2].y, weight[3].y,
                    weight[4].y, weight[5].y, weight[6].y, weight[7].y,
                    array[0], array[1], array[2], array[3],
                    array[4], array[5], array[6], array[7],
                    pos[8].x, pos[8].y, pos[9].x, pos[9].y, pos[10].x, pos[10].y, pos[11].x, pos[11].y,
                    pos[12].x, pos[12].y, pos[13].x, pos[13].y, pos[14].x, pos[14].y, pos[15].x, pos[15].y,
                    tl[8], tl[9], tl[10], tl[11], tl[12], tl[13], tl[14], tl[15],
                    tr[8], tr[9], tr[10], tr[11], tr[12], tr[13], tr[14], tr[15],
                    bl[8], bl[9], bl[10], bl[11], bl[12], bl[13], bl[14], bl[15],
                    br[8], br[9], br[10], br[11], br[12], br[13], br[14], br[15],
                    weight[8].x, weight[9].x, weight[10].x, weight[11].x,
                    weight[12].x, weight[13].x, weight[14].x, weight[15].x,
                    weight[8].y, weight[9].y, weight[10].y, weight[11].y,
                    weight[12].y, weight[13].y, weight[14].y, weight[15].y,
                    array[8], array[9], array[10], array[11],
                    array[12], array[13], array[14], array[15]);
#endif
#if 0   //chroma uv
    XCAM_LOG_ERROR ("\n @ZONGWAVE AVX-CPU interpolate chroma \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tr_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n bl_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n br_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f]; \n interp_avx_uv_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n @ZONGWAVE AVX-CPU \n\n",
                    pos[0].x, pos[0].y, pos[1].x, pos[1].y, pos[2].x, pos[2].y, pos[3].x, pos[3].y,
                    pos[4].x, pos[4].y, pos[5].x, pos[5].y, pos[6].x, pos[6].y, pos[7].x, pos[7].y,
                    tl[0].x, tl[0].y, tl[1].x, tl[1].y, tl[2].x, tl[2].y, tl[3].x, tl[3].y,
                    tl[4].x, tl[4].y, tl[5].x, tl[5].y, tl[6].x, tl[6].y, tl[7].x, tl[7].y,
                    tr[0].x, tr[0].y, tr[1].x, tr[1].y, tr[2].x, tr[2].y, tr[3].x, tr[3].y,
                    tr[4].x, tr[4].y, tr[5].x, tr[5].y, tr[6].x, tr[6].y, tr[7].x, tr[7].y,
                    bl[0].x, bl[0].y, bl[1].x, bl[1].y, bl[2].x, bl[2].y, bl[3].x, bl[3].y,
                    bl[4].x, bl[4].y, bl[5].x, bl[5].y, bl[6].x, bl[6].y, bl[7].x, bl[7].y,
                    br[0].x, br[0].y, br[1].x, br[1].y, br[2].x, br[2].y, br[3].x, br[3].y,
                    br[4].x, br[4].y, br[5].x, br[5].y, br[6].x, br[6].y, br[7].x, br[7].y,
                    weight[0].x, weight[1].x, weight[2].x, weight[3].x,
                    weight[4].x, weight[5].x, weight[6].x, weight[7].x,
                    weight[0].y, weight[1].y, weight[2].y, weight[3].y,
                    weight[4].y, weight[5].y, weight[6].y, weight[7].y,
                    array[0].x, array[0].y, array[1].x, array[1].y,
                    array[2].x, array[2].y, array[3].x, array[3].y,
                    array[4].x, array[4].y, array[5].x, array[5].y,
                    array[6].x, array[6].y, array[7].x, array[7].y);
#endif
}

// interpolate 8 pixels position
// result = p00 * (1 - weight.x) * (1 - weight.y) +
//          p01 * weight.x * (1 - weight.y) +
//          p10 * (1 - weight.x) * weight.y +
//          p11 * weight.x * weight.y;
template <typename T>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, Float2 *array) const
{
    Float2* dest = array;

    __m512 const_one = _mm512_set1_ps (1.0f);
    __m512 const_two = _mm512_set1_ps (2.0f);
    __m512i const_left_idx = _mm512_setr_epi32 (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    __m512i const_right_idx = _mm512_setr_epi32 (2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);
    __m512i const_shuffle = _mm512_setr_epi32 (0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);

    for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS / 8; i++) {
        __m512 interp_pos = _mm512_loadu_ps (&pos[8 * i]);
        __m512 interp_pos_xy = _mm512_floor_ps (interp_pos);
        __m512 interp_weight = _mm512_sub_ps (interp_pos, interp_pos_xy);
        __m512 weight_x = _mm512_permute_ps (interp_weight, 0xA0); //0b10100000
        __m512 weight_y = _mm512_permute_ps (interp_weight, 0xF5); //0b11110101

        int32_t pos_x0 = (int32_t)(pos[8 * i].x);
        border_check_x (pos_x0);

        float* pos_xy = (float*)&interp_pos_xy;
        __m512 pos_00 = _mm512_setr_ps (pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1],
                                        pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1],
                                        pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1],
                                        pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1]);

        __m512i pos_index = _mm512_cvtps_epu32 (_mm512_mul_ps (_mm512_sub_ps (interp_pos_xy, pos_00), const_two));
        __m512i idx_left = _mm512_add_epi32 (_mm512_permutexvar_epi32 (const_shuffle, pos_index), const_left_idx);
        __m512i idx_right = _mm512_add_epi32 (_mm512_permutexvar_epi32 (const_shuffle, pos_index), const_right_idx);

        int32_t pos_y0 = (int32_t)(pos[8 * i].y);
        border_check_y (pos_y0);
        const T* top_ptr = ((const T*)(_buf_ptr + pos_y0 * _pitch));

        int32_t pos_y1 = pos_y0 + 1;
        border_check_y (pos_y1);
        const T* bottom_ptr = ((const T*)(_buf_ptr + pos_y1 * _pitch));

        __m512 tl = _mm512_i32gather_ps (idx_left, (void*) & (top_ptr[pos_x0]), 4);
        __m512 tr = _mm512_i32gather_ps (idx_right, (void*) & (top_ptr[pos_x0]), 4);
        __m512 bl = _mm512_i32gather_ps (idx_left, (void*) & (bottom_ptr[pos_x0]), 4);
        __m512 br = _mm512_i32gather_ps (idx_right, (void*) & (bottom_ptr[pos_x0]), 4);

        __m512 interp_value = _mm512_mul_ps (tl, _mm512_mul_ps (_mm512_sub_ps (const_one, weight_x), _mm512_sub_ps (const_one, weight_y))) +
                              _mm512_mul_ps (tr, _mm512_mul_ps (weight_x, _mm512_sub_ps(const_one, weight_y))) +
                              _mm512_mul_ps (bl, _mm512_mul_ps (_mm512_sub_ps(const_one, weight_x), weight_y)) +
                              _mm512_mul_ps (br, _mm512_mul_ps (weight_x, weight_y));

        _mm512_storeu_ps (dest, interp_value);
        dest += 8;

#if DUMP_AVX_INTERPOLATION
        XCAM_LOG_ERROR ("\n @ZONGWAVE AVX interpolate coordinate \n image_pitch=[%d] \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_avx_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tr_avx_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n bl_avx_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n br_avx_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n interp_avx_coord=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n @ZONGWAVE AVX \n\n",
                        _pitch,
                        ((float*)&interp_pos)[0], ((float*)&interp_pos)[1], ((float*)&interp_pos)[2], ((float*)&interp_pos)[3],
                        ((float*)&interp_pos)[4], ((float*)&interp_pos)[5], ((float*)&interp_pos)[6], ((float*)&interp_pos)[7],
                        ((float*)&interp_pos)[8], ((float*)&interp_pos)[9], ((float*)&interp_pos)[10], ((float*)&interp_pos)[11],
                        ((float*)&interp_pos)[12], ((float*)&interp_pos)[13], ((float*)&interp_pos)[14], ((float*)&interp_pos)[15],
                        ((float*)&tl)[0], ((float*)&tl)[1], ((float*)&tl)[2], ((float*)&tl)[3],
                        ((float*)&tl)[4], ((float*)&tl)[5], ((float*)&tl)[6], ((float*)&tl)[7],
                        ((float*)&tl)[8], ((float*)&tl)[9], ((float*)&tl)[10], ((float*)&tl)[11],
                        ((float*)&tl)[12], ((float*)&tl)[13], ((float*)&tl)[14], ((float*)&tl)[15],
                        ((float*)&tr)[0], ((float*)&tr)[1], ((float*)&tr)[2], ((float*)&tr)[3],
                        ((float*)&tr)[4], ((float*)&tr)[5], ((float*)&tr)[6], ((float*)&tr)[7],
                        ((float*)&tr)[8], ((float*)&tr)[9], ((float*)&tr)[10], ((float*)&tr)[11],
                        ((float*)&tr)[12], ((float*)&tr)[13], ((float*)&tr)[14], ((float*)&tr)[15],
                        ((float*)&bl)[0], ((float*)&bl)[1], ((float*)&bl)[2], ((float*)&bl)[3],
                        ((float*)&bl)[4], ((float*)&bl)[5], ((float*)&bl)[6], ((float*)&bl)[7],
                        ((float*)&bl)[8], ((float*)&bl)[9], ((float*)&bl)[10], ((float*)&bl)[11],
                        ((float*)&bl)[12], ((float*)&bl)[13], ((float*)&bl)[14], ((float*)&bl)[15],
                        ((float*)&br)[0], ((float*)&br)[1], ((float*)&br)[2], ((float*)&br)[3],
                        ((float*)&br)[4], ((float*)&br)[5], ((float*)&br)[6], ((float*)&br)[7],
                        ((float*)&br)[8], ((float*)&br)[9], ((float*)&br)[10], ((float*)&br)[11],
                        ((float*)&br)[12], ((float*)&br)[13], ((float*)&br)[14], ((float*)&br)[15],
                        ((float*)&weight_x)[0], ((float*)&weight_x)[1], ((float*)&weight_x)[2], ((float*)&weight_x)[3],
                        ((float*)&weight_x)[4], ((float*)&weight_x)[5], ((float*)&weight_x)[6], ((float*)&weight_x)[7],
                        ((float*)&weight_x)[8], ((float*)&weight_x)[9], ((float*)&weight_x)[10], ((float*)&weight_x)[11],
                        ((float*)&weight_x)[12], ((float*)&weight_x)[13], ((float*)&weight_x)[14], ((float*)&weight_x)[15],
                        ((float*)&weight_y)[0], ((float*)&weight_y)[1], ((float*)&weight_y)[2], ((float*)&weight_y)[3],
                        ((float*)&weight_y)[4], ((float*)&weight_y)[5], ((float*)&weight_y)[6], ((float*)&weight_y)[7],
                        ((float*)&weight_y)[8], ((float*)&weight_y)[9], ((float*)&weight_y)[10], ((float*)&weight_y)[11],
                        ((float*)&weight_y)[12], ((float*)&weight_y)[13], ((float*)&weight_y)[14], ((float*)&weight_y)[15],
                        array[0].x, array[0].y, array[1].x, array[1].y,
                        array[2].x, array[2].y, array[3].x, array[3].y,
                        array[4].x, array[4].y, array[5].x, array[5].y,
                        array[6].x, array[6].y, array[7].x, array[7].y);
#endif
    }
}

template <typename T>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, Uchar *array) const
{
    float* dest = (float*)array;
    __m256 const_one_float = _mm256_set1_ps (1.0f);
    __m256i const_pitch = _mm256_set1_epi32 (_pitch);
    __m256i const_one_int = _mm256_set1_epi32 (1);
    __m256i index_x = _mm256_setr_epi32 (0, 2, 4, 6, 8, 10, 12, 14);
    __m256i index_y = _mm256_setr_epi32 (1, 3, 5, 7, 9, 11, 13, 15);

    for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS / 8; i++) {
        // load 8 interpolate pos ((8 x float2) x 32bit)
        __m512 interp_pos = _mm512_loadu_ps (&pos[8 * i]);
        __m512 interp_pos_xy = _mm512_floor_ps (interp_pos);
        __m512 interp_weight = _mm512_sub_ps (interp_pos, interp_pos_xy);
        float* weight = (float*)&interp_weight;
        __m256 weight_x = _mm256_i32gather_ps ((float const*)weight, index_x, 4);
        __m256 weight_y = _mm256_i32gather_ps ((float const*)weight, index_y, 4);

        int32_t pos_x0 = (int32_t)(pos[8 * i].x);
        border_check_x (pos_x0);

        float* pos_xy = (float*)&interp_pos_xy;
        __m512 pos_00 = _mm512_setr_ps (0, pos_xy[1], 0, pos_xy[1],
                                        0, pos_xy[1], 0, pos_xy[1],
                                        0, pos_xy[1], 0, pos_xy[1],
                                        0, pos_xy[1], 0, pos_xy[1]);

        __m512i pos_index = _mm512_cvtps_epi32 (_mm512_sub_ps (interp_pos_xy, pos_00));
        int32_t* idx = (int32_t*) &pos_index;
        __m256i pos_idx_x = _mm256_i32gather_epi32 ((int const*)idx, index_x, 4);
        __m256i pos_idx_y = _mm256_i32gather_epi32 ((int const*)idx, index_y, 4);

        int32_t pos_y0 = (int32_t)(pos[8 * i].y);
        border_check_y (pos_y0);
        const T* base = ((const T*)(_buf_ptr + pos_y0 * _pitch));

        __m256i offset_top0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (pos_idx_y, const_pitch));
        int32_t* offset_tl = (int32_t*)&offset_top0;

        __m256i offset_top1 = _mm256_add_epi32 (offset_top0, const_one_int);
        int32_t* offset_tr = (int32_t*)&offset_top1;

        __m256i offset_bottom0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (_mm256_add_epi32(pos_idx_y, const_one_int), const_pitch));
        int32_t* offset_bl = (int32_t*)&offset_bottom0;

        __m256i offset_bottom1 = _mm256_add_epi32 (offset_bottom0, const_one_int);
        int32_t* offset_br = (int32_t*)&offset_bottom1;

        __m256 pixel_tl = _mm256_setr_ps (*(base + offset_tl[0]), *(base + offset_tl[1]), *(base + offset_tl[2]), *(base + offset_tl[3]),
                                          *(base + offset_tl[4]), *(base + offset_tl[5]), *(base + offset_tl[6]), *(base + offset_tl[7]));
        __m256 pixel_tr = _mm256_setr_ps (*(base + offset_tr[0]), *(base + offset_tr[1]), *(base + offset_tr[2]), *(base + offset_tr[3]),
                                          *(base + offset_tr[4]), *(base + offset_tr[5]), *(base + offset_tr[6]), *(base + offset_tr[7]));
        __m256 pixel_bl = _mm256_setr_ps (*(base + offset_bl[0]), *(base + offset_bl[1]), *(base + offset_bl[2]), *(base + offset_bl[3]),
                                          *(base + offset_bl[4]), *(base + offset_bl[5]), *(base + offset_bl[6]), *(base + offset_bl[7]));
        __m256 pixel_br = _mm256_setr_ps (*(base + offset_br[0]), *(base + offset_br[1]), *(base + offset_br[2]), *(base + offset_br[3]),
                                          *(base + offset_br[4]), *(base + offset_br[5]), *(base + offset_br[6]), *(base + offset_br[7]));

        __m256 interp_value_f = _mm256_mul_ps (pixel_tl, _mm256_mul_ps (_mm256_sub_ps (const_one_float, weight_x), _mm256_sub_ps (const_one_float, weight_y))) +
                                _mm256_mul_ps (pixel_tr, _mm256_mul_ps (weight_x, _mm256_sub_ps(const_one_float, weight_y))) +
                                _mm256_mul_ps (pixel_bl, _mm256_mul_ps (_mm256_sub_ps(const_one_float, weight_x), weight_y)) +
                                _mm256_mul_ps (pixel_br, _mm256_mul_ps (weight_x, weight_y));

        interp_value_f = _mm256_round_ps (interp_value_f, _MM_FROUND_TO_NEAREST_INT);
        __m256i interp_value = _mm256_cvtps_epi32 (interp_value_f);
        interp_value = _mm256_packs_epi32 (interp_value, interp_value);
        interp_value = _mm256_packus_epi16 (interp_value, interp_value);

        _mm_store_ss (dest, (__m128)_mm256_extractf128_si256 (interp_value, 0));
        _mm_store_ss (dest + 1, (__m128)_mm256_extractf128_si256 (interp_value, 1));
        dest += 2;

#if DUMP_AVX_INTERPOLATION
        XCAM_LOG_ERROR ("\n @ZONGWAVE AVX interpolate luma \n image_pitch=[%d] \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_avx_pixel=[%f %f %f %f %f %f %f %f]; \n tr_avx_pixel=[%f %f %f %f %f %f %f %f]; \n bl_avx_pixel=[%f %f %f %f %f %f %f %f]; \n br_avx_pixel=[%f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f]; \n interp_avx_y_pixel=[%d %d %d %d %d %d %d %d]; \n @ZONGWAVE AVX \n\n",
                        _pitch,
                        ((float*)&interp_pos)[0], ((float*)&interp_pos)[1], ((float*)&interp_pos)[2], ((float*)&interp_pos)[3],
                        ((float*)&interp_pos)[4], ((float*)&interp_pos)[5], ((float*)&interp_pos)[6], ((float*)&interp_pos)[7],
                        ((float*)&interp_pos)[8], ((float*)&interp_pos)[9], ((float*)&interp_pos)[10], ((float*)&interp_pos)[11],
                        ((float*)&interp_pos)[12], ((float*)&interp_pos)[13], ((float*)&interp_pos)[14], ((float*)&interp_pos)[15],
                        ((float*)&pixel_tl)[0], ((float*)&pixel_tl)[1], ((float*)&pixel_tl)[2], ((float*)&pixel_tl)[3],
                        ((float*)&pixel_tl)[4], ((float*)&pixel_tl)[5], ((float*)&pixel_tl)[6], ((float*)&pixel_tl)[7],
                        ((float*)&pixel_tr)[0], ((float*)&pixel_tr)[1], ((float*)&pixel_tr)[2], ((float*)&pixel_tr)[3],
                        ((float*)&pixel_tr)[4], ((float*)&pixel_tr)[5], ((float*)&pixel_tr)[6], ((float*)&pixel_tr)[7],
                        ((float*)&pixel_bl)[0], ((float*)&pixel_bl)[1], ((float*)&pixel_bl)[2], ((float*)&pixel_bl)[3],
                        ((float*)&pixel_bl)[4], ((float*)&pixel_bl)[5], ((float*)&pixel_bl)[6], ((float*)&pixel_bl)[7],
                        ((float*)&pixel_br)[0], ((float*)&pixel_br)[1], ((float*)&pixel_br)[2], ((float*)&pixel_br)[3],
                        ((float*)&pixel_br)[4], ((float*)&pixel_br)[5], ((float*)&pixel_br)[6], ((float*)&pixel_br)[7],
                        ((float*)&weight_x)[0], ((float*)&weight_x)[1], ((float*)&weight_x)[2], ((float*)&weight_x)[3],
                        ((float*)&weight_x)[4], ((float*)&weight_x)[5], ((float*)&weight_x)[6], ((float*)&weight_x)[7],
                        ((float*)&weight_y)[0], ((float*)&weight_y)[1], ((float*)&weight_y)[2], ((float*)&weight_y)[3],
                        ((float*)&weight_y)[4], ((float*)&weight_y)[5], ((float*)&weight_y)[6], ((float*)&weight_y)[7],
                        array[0], array[1], array[2], array[3],
                        array[4], array[5], array[6], array[7]);
#endif
    }
}

template <typename T>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, Uchar2 *array) const
{
    float* dest = (float*)array;

    __m512 const_one_float = _mm512_set1_ps (1.0f);
    __m256i const_pitch = _mm256_set1_epi32 (_pitch / 2);
    __m256i const_one_int = _mm256_set1_epi32 (1);
    __m256i index_x = _mm256_setr_epi32 (0, 2, 4, 6, 8, 10, 12, 14);
    __m256i index_y = _mm256_setr_epi32 (1, 3, 5, 7, 9, 11, 13, 15);

    // load 8 interpolate pos ((8 x float2) x 32bit)
    __m512 interp_pos = _mm512_loadu_ps (pos);

    __m512 interp_pos_xy = _mm512_floor_ps (interp_pos);
    __m512 interp_weight = _mm512_sub_ps (interp_pos, interp_pos_xy);

    __m512 weight_x = _mm512_permute_ps (interp_weight, 0xA0); //0b10100000
    __m512 weight_y = _mm512_permute_ps (interp_weight, 0xF5); //0b11110101

    int32_t pos_x0 = (int32_t)(pos[0].x);
    border_check_x (pos_x0);

    float* pos_xy = (float*)&interp_pos_xy;
    __m512 pos_00 = _mm512_setr_ps (0, pos_xy[1], 0, pos_xy[1],
                                    0, pos_xy[1], 0, pos_xy[1],
                                    0, pos_xy[1], 0, pos_xy[1],
                                    0, pos_xy[1], 0, pos_xy[1]);

    __m512i pos_index = _mm512_cvtps_epi32 (_mm512_sub_ps (interp_pos_xy, pos_00));
    int32_t* idx = (int32_t*) &pos_index;
    __m256i pos_idx_x = _mm256_i32gather_epi32 ((int const*)idx, index_x, 4);
    __m256i pos_idx_y = _mm256_i32gather_epi32 ((int const*)idx, index_y, 4);

    int32_t pos_y0 = (int32_t)(pos[0].y);
    border_check_y (pos_y0);

    int32_t pos_y1 = pos_y0 + 1;
    border_check_y (pos_y1);

    const T* base = ((const T*)(_buf_ptr + pos_y0 * _pitch));
    __m256i offset_top0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (pos_idx_y, const_pitch));
    int32_t* offset_tl = (int32_t*)&offset_top0;

    __m256i offset_top1 = _mm256_add_epi32 (offset_top0, const_one_int);
    int32_t* offset_tr = (int32_t*)&offset_top1;

    __m256i offset_bottom0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (_mm256_add_epi32(pos_idx_y, const_one_int), const_pitch));
    int32_t* offset_bl = (int32_t*)&offset_bottom0;

    __m256i offset_bottom1 = _mm256_add_epi32 (offset_bottom0, const_one_int);
    int32_t* offset_br = (int32_t*)&offset_bottom1;

    __m512 pixel_tl = _mm512_setr_ps ((base + offset_tl[0])->x, (base + offset_tl[0])->y, (base + offset_tl[1])->x, (base + offset_tl[1])->y,
                                      (base + offset_tl[2])->x, (base + offset_tl[2])->y, (base + offset_tl[3])->x, (base + offset_tl[3])->y,
                                      (base + offset_tl[4])->x, (base + offset_tl[4])->y, (base + offset_tl[5])->x, (base + offset_tl[5])->y,
                                      (base + offset_tl[6])->x, (base + offset_tl[6])->y, (base + offset_tl[7])->x, (base + offset_tl[7])->y);

    __m512 pixel_tr = _mm512_setr_ps ((base + offset_tr[0])->x, (base + offset_tr[0])->y, (base + offset_tr[1])->x, (base + offset_tr[1])->y,
                                      (base + offset_tr[2])->x, (base + offset_tr[2])->y, (base + offset_tr[3])->x, (base + offset_tr[3])->y,
                                      (base + offset_tr[4])->x, (base + offset_tr[4])->y, (base + offset_tr[5])->x, (base + offset_tr[5])->y,
                                      (base + offset_tr[6])->x, (base + offset_tr[6])->y, (base + offset_tr[7])->x, (base + offset_tr[7])->y);
    __m512 pixel_bl = _mm512_setr_ps ((base + offset_bl[0])->x, (base + offset_bl[0])->y, (base + offset_bl[1])->x, (base + offset_bl[1])->y,
                                      (base + offset_bl[2])->x, (base + offset_bl[2])->y, (base + offset_bl[3])->x, (base + offset_bl[3])->y,
                                      (base + offset_bl[4])->x, (base + offset_bl[4])->y, (base + offset_bl[5])->x, (base + offset_bl[5])->y,
                                      (base + offset_bl[6])->x, (base + offset_bl[6])->y, (base + offset_bl[7])->x, (base + offset_bl[7])->y);
    __m512 pixel_br = _mm512_setr_ps ((base + offset_br[0])->x, (base + offset_br[0])->y, (base + offset_br[1])->x, (base + offset_br[1])->y,
                                      (base + offset_br[2])->x, (base + offset_br[2])->y, (base + offset_br[3])->x, (base + offset_br[3])->y,
                                      (base + offset_br[4])->x, (base + offset_br[4])->y, (base + offset_br[5])->x, (base + offset_br[5])->y,
                                      (base + offset_br[6])->x, (base + offset_br[6])->y, (base + offset_br[7])->x, (base + offset_br[7])->y);

    __m512 interp_value_f = _mm512_mul_ps (pixel_tl, _mm512_mul_ps (_mm512_sub_ps (const_one_float, weight_x), _mm512_sub_ps (const_one_float, weight_y))) +
                            _mm512_mul_ps (pixel_tr, _mm512_mul_ps (weight_x, _mm512_sub_ps(const_one_float, weight_y))) +
                            _mm512_mul_ps (pixel_bl, _mm512_mul_ps (_mm512_sub_ps(const_one_float, weight_x), weight_y)) +
                            _mm512_mul_ps (pixel_br, _mm512_mul_ps (weight_x, weight_y));

    //interp_value_f = _mm512_round_ps (interp_value_f, _MM_FROUND_TO_NEAREST_INT);

    __m512i interp_value = _mm512_cvtps_epi32 (interp_value_f);
    interp_value = _mm512_packs_epi32 (interp_value, interp_value);
    interp_value = _mm512_packus_epi16 (interp_value, interp_value);

    _mm_store_ss (dest,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 0));
    _mm_store_ss (dest + 1,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 1));
    _mm_store_ss (dest + 2,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 2));
    _mm_store_ss (dest + 3,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 3));

#if DUMP_AVX_INTERPOLATION
    XCAM_LOG_ERROR ("\n @ZONGWAVE AVX interpolate chroma \n image_pitch=[%d] \n lut_pos=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tl_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n tr_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n bl_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n br_avx_pixel=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n weight_x=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n weight_y=[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]; \n interp_avx_uv_pixel=[%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d];  \n @ZONGWAVE AVX \n\n",
                    _pitch,
                    ((float*)&interp_pos)[0], ((float*)&interp_pos)[1], ((float*)&interp_pos)[2], ((float*)&interp_pos)[3],
                    ((float*)&interp_pos)[4], ((float*)&interp_pos)[5], ((float*)&interp_pos)[6], ((float*)&interp_pos)[7],
                    ((float*)&interp_pos)[8], ((float*)&interp_pos)[9], ((float*)&interp_pos)[10], ((float*)&interp_pos)[11],
                    ((float*)&interp_pos)[12], ((float*)&interp_pos)[13], ((float*)&interp_pos)[14], ((float*)&interp_pos)[15],
                    ((float*)&pixel_tl)[0], ((float*)&pixel_tl)[1], ((float*)&pixel_tl)[2], ((float*)&pixel_tl)[3],
                    ((float*)&pixel_tl)[4], ((float*)&pixel_tl)[5], ((float*)&pixel_tl)[6], ((float*)&pixel_tl)[7],
                    ((float*)&pixel_tl)[8], ((float*)&pixel_tl)[9], ((float*)&pixel_tl)[10], ((float*)&pixel_tl)[11],
                    ((float*)&pixel_tl)[12], ((float*)&pixel_tl)[13], ((float*)&pixel_tl)[14], ((float*)&pixel_tl)[15],
                    ((float*)&pixel_tr)[0], ((float*)&pixel_tr)[1], ((float*)&pixel_tr)[2], ((float*)&pixel_tr)[3],
                    ((float*)&pixel_tr)[4], ((float*)&pixel_tr)[5], ((float*)&pixel_tr)[6], ((float*)&pixel_tr)[7],
                    ((float*)&pixel_tr)[8], ((float*)&pixel_tr)[9], ((float*)&pixel_tr)[10], ((float*)&pixel_tr)[11],
                    ((float*)&pixel_tr)[12], ((float*)&pixel_tr)[13], ((float*)&pixel_tr)[14], ((float*)&pixel_tr)[15],
                    ((float*)&pixel_bl)[0], ((float*)&pixel_bl)[1], ((float*)&pixel_bl)[2], ((float*)&pixel_bl)[3],
                    ((float*)&pixel_bl)[4], ((float*)&pixel_bl)[5], ((float*)&pixel_bl)[6], ((float*)&pixel_bl)[7],
                    ((float*)&pixel_bl)[8], ((float*)&pixel_bl)[9], ((float*)&pixel_bl)[10], ((float*)&pixel_bl)[11],
                    ((float*)&pixel_bl)[12], ((float*)&pixel_bl)[13], ((float*)&pixel_bl)[14], ((float*)&pixel_bl)[15],
                    ((float*)&pixel_br)[0], ((float*)&pixel_br)[1], ((float*)&pixel_br)[2], ((float*)&pixel_br)[3],
                    ((float*)&pixel_br)[4], ((float*)&pixel_br)[5], ((float*)&pixel_br)[6], ((float*)&pixel_br)[7],
                    ((float*)&pixel_br)[8], ((float*)&pixel_br)[9], ((float*)&pixel_br)[10], ((float*)&pixel_br)[11],
                    ((float*)&pixel_br)[12], ((float*)&pixel_br)[13], ((float*)&pixel_br)[14], ((float*)&pixel_br)[15],
                    ((float*)&weight_x)[0], ((float*)&weight_x)[1], ((float*)&weight_x)[2], ((float*)&weight_x)[3],
                    ((float*)&weight_x)[4], ((float*)&weight_x)[5], ((float*)&weight_x)[6], ((float*)&weight_x)[7],
                    ((float*)&weight_x)[8], ((float*)&weight_x)[9], ((float*)&weight_x)[10], ((float*)&weight_x)[11],
                    ((float*)&weight_x)[12], ((float*)&weight_x)[13], ((float*)&weight_x)[14], ((float*)&weight_x)[15],
                    ((float*)&weight_y)[0], ((float*)&weight_y)[1], ((float*)&weight_y)[2], ((float*)&weight_y)[3],
                    ((float*)&weight_y)[4], ((float*)&weight_y)[5], ((float*)&weight_y)[6], ((float*)&weight_y)[7],
                    ((float*)&weight_y)[8], ((float*)&weight_y)[9], ((float*)&weight_y)[10], ((float*)&weight_y)[11],
                    ((float*)&weight_y)[12], ((float*)&weight_y)[13], ((float*)&weight_y)[14], ((float*)&weight_y)[15],
                    array[0].x, array[0].y, array[1].x, array[1].y, array[2].x, array[2].y, array[3].x, array[3].y,
                    array[4].x, array[4].y, array[5].x, array[5].y, array[6].x, array[6].y, array[7].x, array[7].y);
#endif
}

#endif

}
#endif //XCAM_SOFT_IMAGE_H
