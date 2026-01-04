/*
 *  Copyright 2010-2011 Alessandro Francescon
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
 */

#include "zxing/MatSource.h"
#include <zxing/common/IllegalArgumentException.h>

zxing::Ref<zxing::LuminanceSource> MatSource::create(const ::cv::Mat& cvImage) {
    return zxing::Ref<LuminanceSource>(new MatSource(cvImage));
}

MatSource::MatSource(const ::cv::Mat& _cvImage) : zxing::LuminanceSource(_cvImage.cols, _cvImage.rows) {
    cvImage = _cvImage.clone();
}

zxing::ArrayRef<char> MatSource::getRow(int y, zxing::ArrayRef<char> row) const {
    int width = getWidth();

    if (!row) {
        row = zxing::ArrayRef<char>(width);
    }

    const char* p = cvImage.ptr<char>(y);

    for (int x = 0; x < width; ++x, ++p) {
        row[x] = *p;
    }

    return row;
}

zxing::ArrayRef<char> MatSource::getMatrix() const {
    int width = getWidth();
    int height = getHeight();

    zxing::ArrayRef<char> matrix = zxing::ArrayRef<char>(width * height);

    for (int y = 0; y < height; ++y) {
        const char* p = cvImage.ptr<char>(y);
        int yoffset = y * width;

        for (int x = 0; x < width; ++x, ++p) {
            matrix[yoffset + x] = *p;
        }
    }

    return matrix;
}
