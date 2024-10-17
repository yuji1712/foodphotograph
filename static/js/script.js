document.addEventListener("DOMContentLoaded", function () {
  // 画像が選択されたときに、アップロードエリアの画像を変更
  document.getElementById("imageInput").addEventListener("change", function () {
    const uploadedImage = this.files[0];
    if (uploadedImage) {
      const uploadedImageURL = URL.createObjectURL(uploadedImage);
      document.getElementById("clickableImage").src = uploadedImageURL;
    }
  });

  document.getElementById("uploadForm").onsubmit = async function (e) {
    e.preventDefault();
    const formData = new FormData(this);

    // 画像をサーバーに送信
    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("受け取ったデータ:", data); // デバッグ用

        const results = data.results;
        console.log("結果オブジェクト:", results); // デバッグ用

        // 結果メッセージを表示
        document.getElementById("resultMessage").innerHTML = `
                <div class="result-card">
                <h3>分析結果</h3>
                <p><strong>構図:</strong> ${results.composition}</p>
                <p><strong>角度:</strong> ${results.angle}</p>
                <p><strong>シズル効果（光沢）:</strong> ${results.sizzle_shiny}</p>
                <p><strong>シズル効果（動き）:</strong> ${results.sizzle_motion}</p>
                <p><strong>シズル効果（蒸気）:</strong> ${results.sizzle_steam}</p>
                <p><strong>スコア:</strong> ${results.score}</p> <!-- スコアを追加 -->
                </div>
            `;
      } else {
        console.error("Response not OK:", response.status);
        document.getElementById("resultMessage").textContent =
          "エラーが発生しました。";
      }
    } catch (error) {
      console.error("Fetch error:", error);
      document.getElementById("resultMessage").textContent =
        "エラーが発生しました。";
    }
  };
});
