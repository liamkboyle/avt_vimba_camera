/// Copyright (c) 2014,
/// Systems, Robotics and Vision Group
/// University of the Balearic Islands
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///     * All advertising materials mentioning features or use of this software
///       must display the following acknowledgement:
///       This product includes software developed by
///       Systems, Robotics and Vision Group, Univ. of the Balearic Islands
///     * Neither the name of Systems, Robotics and Vision Group, University of
///       the Balearic Islands nor the names of its contributors may be used
///       to endorse or promote products derived from this software without
///       specific prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
/// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
/// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
/// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
/// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
/// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <thread>

#include <avt_vimba_camera/mono_camera_node.hpp>
#include <avt_vimba_camera_msgs/srv/load_settings.hpp>
#include <avt_vimba_camera_msgs/srv/save_settings.hpp>

#include <VimbaCPP/Include/VimbaCPP.h>

#include <opencv2/core.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

#include <immintrin.h>
#include <tiffio.h>
#include <string>

using namespace std::placeholders;

namespace avt_vimba_camera
{
MonoCameraNode::MonoCameraNode() : Node("camera"), api_(this->get_logger()), cam_(std::shared_ptr<rclcpp::Node>(dynamic_cast<rclcpp::Node * >(this)))
{
  // Set the image publisher before streaming
  camera_info_pub_ = image_transport::create_camera_publisher(this, "~/image", rmw_qos_profile_system_default);


  std::cout << cv::getBuildInformation() << std::endl;
  write_tiffs_ = false;
  // Set the frame callback
  cam_.setCallback(std::bind(&avt_vimba_camera::MonoCameraNode::frameCallback, this, _1));

  start_srv_ = create_service<std_srvs::srv::Trigger>("~/start_stream", std::bind(&MonoCameraNode::startSrvCallback, this, _1, _2, _3));
  stop_srv_ = create_service<std_srvs::srv::Trigger>("~/stop_stream", std::bind(&MonoCameraNode::stopSrvCallback, this, _1, _2, _3));

  start_writing_ = create_service<std_srvs::srv::Trigger>("~/start_writing_tiffs", std::bind(&MonoCameraNode::start_writing_tiffs, this, _1, _2, _3));
  stop_writing_ = create_service<std_srvs::srv::Trigger>("~/stop_writing_tiffs", std::bind(&MonoCameraNode::stop_writing_tiffs, this, _1, _2, _3));



  // load_srv_ = create_service<avt_vimba_camera_msgs::srv::LoadSettings>("~/load_settings", std::bind(&MonoCameraNode::loadSrvCallback, this, _1, _2, _3));
  // save_srv_ = create_service<avt_vimba_camera_msgs::srv::SaveSettings>("~/save_settings", std::bind(&MonoCameraNode::saveSrvCallback, this, _1, _2, _3));

  loadParams();
}

MonoCameraNode::~MonoCameraNode()
{
  cam_.stop();
  camera_info_pub_.shutdown();
}

void MonoCameraNode::loadParams()
{
  ip_ = this->declare_parameter("ip", "");
  guid_ = this->declare_parameter("guid", "");
  camera_info_url_ = this->declare_parameter("camera_info_url", "");
  frame_id_ = this->declare_parameter("frame_id", "");
  use_measurement_time_ = this->declare_parameter("use_measurement_time", false);
  ptp_offset_ = this->declare_parameter("ptp_offset", 0);
  start_imaging_ = this->declare_parameter("start_imaging", false);
  camera_name_ = this->declare_parameter("name", "mono_camera");

  RCLCPP_INFO(this->get_logger(), "Parameters loaded");
}

void MonoCameraNode::start()
{
  // Start Vimba & list all available cameras
  api_.start();

  // Start camera
  cam_.start(ip_, guid_, frame_id_, camera_info_url_);

  // Start imaging on camera Node start
  cam_.startImaging();
  
  // // Pause to allow the camera to start
  // if(!start_imaging_){
  //   cam_.stopImaging();
  //   cam_.setForceStop(true);
  // }
}



bool simd_unpack12to16(uint16_t *out_ptr, uint8_t *in_ptr, size_t n){
  

    const __m256i bytegrouping =
        _mm256_setr_epi8(4,5, 5,6,  7,8, 8,9,  10,11, 11,12,  13,14, 14,15, // low half uses last 12B
                         0,1, 1,2,  3,4, 4,5,   6, 7,  7, 8,   9,10, 10,11); // high half uses first 12B
    

    // First Round
    __m256i v = _mm256_loadu_si256((__m256i*)in_ptr);
    v = _mm256_permutevar8x32_epi32(v, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0));
    
    v = _mm256_shuffle_epi8(v, bytegrouping);


    __m256i hi = _mm256_srli_epi16(v, 4);                              // [ 0 f e d | xxxx ]
    __m256i lo  = _mm256_and_si256(v, _mm256_set1_epi32(0x00000FFF));

    v = _mm256_blend_epi16(lo, hi, 0b10101010);
    v = _mm256_slli_epi16(v, 4);

    _mm256_storeu_si256((__m256i*)out_ptr, v);
    out_ptr += 16;
    in_ptr += (24 - 4);
    n -= 24;



    while(n){

        __m256i v = _mm256_loadu_si256((__m256i*)in_ptr);
        v = _mm256_shuffle_epi8(v, bytegrouping);


        __m256i hi = _mm256_srli_epi16(v, 4);                              // [ 0 f e d | xxxx ]
        __m256i lo  = _mm256_and_si256(v, _mm256_set1_epi32(0x00000FFF));

        v = _mm256_blend_epi16(lo, hi, 0b10101010);
        v = _mm256_slli_epi16(v, 4);

        _mm256_storeu_si256((__m256i*)out_ptr, v);

        out_ptr += 16;
        in_ptr += 24;
        n-= 24;
    }

    return true;

}


void WriteImage(cv::Mat& image, const char* filename)
{
	TIFF* tiff_img = TIFFOpen(filename, "w");
	if (!tiff_img)
		throw "Opening image by TIFFOpen";

	// set tiff tags
	int width = image.cols;
	int height = image.rows; 


	int bits_per_sample = 8;
  if(image.type() == CV_16U || image.type() == CV_16UC1 || image.type () == CV_16UC3){
      bits_per_sample = 16;
  }

  int line_len = 0;
	line_len = width * (bits_per_sample / 8);
	// printf("Saving image %dx%d to file:%s\n", width, height, filename);

	TIFFSetField(tiff_img, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tiff_img, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tiff_img, TIFFTAG_ROWSPERSTRIP, height);
	
  TIFFSetField(tiff_img, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
	TIFFSetField(tiff_img, TIFFTAG_MINSAMPLEVALUE, 0);
	TIFFSetField(tiff_img, TIFFTAG_MAXSAMPLEVALUE, (1 << bits_per_sample) - 1);

	TIFFSetField(tiff_img, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tiff_img, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

	if (image.type() == CV_16UC3 || image.type() == CV_8UC3){
      TIFFSetField(tiff_img, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	    TIFFSetField(tiff_img, TIFFTAG_SAMPLESPERPIXEL, 3);
  } else {
      TIFFSetField(tiff_img, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	    TIFFSetField(tiff_img, TIFFTAG_SAMPLESPERPIXEL, 1);
  }
	//TIFFSetField(tiff_img, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	//TIFFSetField(tiff_img, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);

	// save data
	if (TIFFWriteEncodedStrip(tiff_img, 0, image.data, line_len*height) == -1)
	{
		throw("ImageFailed to write image");
	}

	TIFFWriteDirectory(tiff_img);
	TIFFClose(tiff_img);
}



bool get_raw_and_thumbnail(const FramePtr vimba_frame_ptr, cv::Mat &raw_img, sensor_msgs::msg::Image& image)
{
    VmbPixelFormatType pixel_format;
    VmbUint32_t width, height, nSize;
    VmbUint64_t ts;

    vimba_frame_ptr->GetWidth(width);
    vimba_frame_ptr->GetHeight(height);
    vimba_frame_ptr->GetPixelFormat(pixel_format);
    vimba_frame_ptr->GetImageSize(nSize);
    vimba_frame_ptr->GetTimestamp(ts);
    VmbUchar_t* buffer_ptr;
    VmbErrorType err = vimba_frame_ptr->GetImage(buffer_ptr);
    if(err != VmbErrorSuccess){
      return false;
    }


    // std::cout << width << "x" << height << " ts:" << ts << " pxl_fmt:" << pixel_format << " n_size:" << nSize << std::endl;
    cv::Mat raw_8bit;
    cv::Mat resized_down;
    cv_bridge::CvImage cv_img;


    switch(pixel_format){


      case VmbPixelFormatRgb8: 
        raw_img = cv::Mat(height, width, CV_8UC3, buffer_ptr);
        raw_8bit = raw_img;
        // cv::cvtColor(orig_image, orig_image, cv::COLOR_RGB2GRAY);
        // cv_img.encoding = "mono8";
        cv_img.encoding = "rgb8";

        break;

      case VmbPixelFormatBayerRG12p:
        {
          raw_img = cv::Mat(height, width, CV_16UC1);

          uint16_t *img_ptr = raw_img.ptr<uint16_t>();

          simd_unpack12to16(img_ptr, buffer_ptr, nSize);

          raw_img.convertTo(raw_8bit, CV_8U, 255.0 / 65535);
          cv::cvtColor(raw_8bit, raw_8bit, cv::COLOR_BayerRG2BGR);
        }
        cv_img.encoding = "rgb8";
        break;


      default:
        raw_img = cv::Mat(height, width, CV_8UC3, buffer_ptr);
        raw_8bit - raw_img;
        cv_img.encoding = "rgb8";
        break;
    }


    

    cv::Size newSize(raw_8bit.cols / 8, raw_8bit.rows / 8);
    cv::resize(raw_8bit, resized_down, newSize, cv::INTER_LINEAR);
    // cv::imshow("resiresize_down);

    // std::cout << cvImage << std::endl;

    cv_img.image = resized_down;

    cv_img.toImageMsg(image);

    return true;
}




void MonoCameraNode::frameCallback(const FramePtr& vimba_frame_ptr)
{
  rclcpp::Time ros_time = this->get_clock()->now();
  VmbUint64_t ts;
  VmbUint64_t frame_id;
  vimba_frame_ptr->GetTimestamp(ts);
  vimba_frame_ptr->GetFrameID(frame_id);

  // std::cout << "[" << camera_name_ << "] Frame ID: " << frame_id << std::endl;

  // getNumSubscribers() is not yet supported in Foxy, will be supported in later versions
  // if (camera_info_pub_.getNumSubscribers() > 0)
  {
    sensor_msgs::msg::Image img;
    cv::Mat raw_img;



    if(get_raw_and_thumbnail(vimba_frame_ptr, raw_img, img))
    {


      {
        std::unique_lock<std::mutex> lock(write_tiffs_mutex_);

        if(write_tiffs_){
          std::string filename = "/data/raw_" + camera_name_ + "_" + std::to_string(frame_id) + "_" + std::to_string(ts) + ".tiff";
          WriteImage(raw_img, filename.c_str());
        }
      }

      sensor_msgs::msg::CameraInfo ci = cam_.getCameraInfo();
      // Note: getCameraInfo() doesn't fill in header frame_id or stamp
      ci.header.frame_id = frame_id_;
      if (use_measurement_time_)
      {
        VmbUint64_t frame_timestamp;
        vimba_frame_ptr->GetTimestamp(frame_timestamp);
        ci.header.stamp = rclcpp::Time(cam_.getTimestampRealTime(frame_timestamp)) + rclcpp::Duration(ptp_offset_, 0);
      }
      else
      {
        ci.header.stamp = ros_time;
      }
      img.header.frame_id = ci.header.frame_id;
      img.header.stamp = ci.header.stamp;
      
      camera_info_pub_.publish(img, ci);
    }
    else
    {
      RCLCPP_WARN_STREAM(this->get_logger(), "Function frameToImage returned 0. No image published.");
    }
  }
}

void MonoCameraNode::start_writing_tiffs(const std::shared_ptr<rmw_request_id_t> request_header,
                                      const std_srvs::srv::Trigger::Request::SharedPtr req,
                                      std_srvs::srv::Trigger::Response::SharedPtr res) {
  (void)request_header;
  (void)req;
  std::unique_lock<std::mutex> lock(write_tiffs_mutex_);
  
  write_tiffs_ = true;
  res->success = true;
}

void MonoCameraNode::stop_writing_tiffs(const std::shared_ptr<rmw_request_id_t> request_header,
                                      const std_srvs::srv::Trigger::Request::SharedPtr req,
                                      std_srvs::srv::Trigger::Response::SharedPtr res) {
  (void)request_header;
  (void)req;
  std::unique_lock<std::mutex> lock(write_tiffs_mutex_);

  write_tiffs_ = false;
  res->success = true;
}


void MonoCameraNode::startSrvCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                                      const std_srvs::srv::Trigger::Request::SharedPtr req,
                                      std_srvs::srv::Trigger::Response::SharedPtr res) {
  (void)request_header;
  (void)req;

  cam_.startImaging();
  cam_.setForceStop(false);
  auto state = cam_.getCameraState();
  res->success = state != CameraState::ERROR;
}

void MonoCameraNode::stopSrvCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                                     const std_srvs::srv::Trigger::Request::SharedPtr req,
                                     std_srvs::srv::Trigger::Response::SharedPtr res)
{
  (void)request_header;
  (void)req;

  cam_.stopImaging();
  cam_.setForceStop(true);
  auto state = cam_.getCameraState();
  res->success = state != CameraState::ERROR;
}

void MonoCameraNode::loadSrvCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                                     const avt_vimba_camera_msgs::srv::LoadSettings::Request::SharedPtr req,
                                     avt_vimba_camera_msgs::srv::LoadSettings::Response::SharedPtr res) 
{
  (void)request_header;
  auto extension = req->input_path.substr(req->input_path.find_last_of(".") + 1);
  if (extension != "xml")
  {
    RCLCPP_WARN(this->get_logger(), "Invalid file extension. Only .xml is supported.");
    res->result = false;
  } 
  else 
  {
    res->result = cam_.loadCameraSettings(req->input_path);
  }
}

void MonoCameraNode::saveSrvCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                                     const avt_vimba_camera_msgs::srv::SaveSettings::Request::SharedPtr req,
                                     avt_vimba_camera_msgs::srv::SaveSettings::Response::SharedPtr res) 
{
  (void)request_header;
  auto extension = req->output_path.substr(req->output_path.find_last_of(".") + 1);
  if (extension != "xml")
  {
    RCLCPP_WARN(this->get_logger(), "Invalid file extension. Only .xml is supported.");
    res->result = false;
  } 
  else 
  {
    res->result = cam_.saveCameraSettings(req->output_path);
  }
}
}  // namespace avt_vimba_camera
