cv_template_matching_sample
====
OpenCVを使ったテンプレートマッチングの簡単なサンプル。

<pre>
   double    match_threshold = 0.7;
   cv::Point match_pos;
   bool      has_match_pos;
        ・
        ・
        ・
    while(true) {
        capture >> capture_img;

        has_match_pos = false;
        
        if (!template_img.empty()) {
            // グレースケール画像に変換
            cv::Mat gray_img;
            cv::cvtColor(capture_img, gray_img, CV_BGR2GRAY);

            // 結果格納用cv::Matの準備
            cv::Size result_size(gray_img.cols - template_img.cols + 1, gray_img.rows - template_img.rows + 1)
            cv::Mat result(result_size, CV_32FC1);

            // テンプレートマッチングの実行
            cv::matchTemplate(gray_img, template_img, result, CV_TM_CCOEFF_NORMED);

            // CV_TM_CCOEFF_NORMEDの場合は最大値の座標がマッチ位置
            double min_val, max_val, match_val;
            cv::Point min_pos, max_pos;
            cv::minMaxLoc(result, &min_val, &max_val, &min_pos, &max_pos, cv::Mat());

            cv::Point match_pos = max_pos;
            double    match_val = max_val;

            // 設定した閾値以上の場合はマッチしていると判断する
            if (match_val > match_threshold) {
                has_match_pos = true;
            }
        }
</pre>
