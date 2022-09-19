const BASE_URL = "http://127.0.0.9:4455";

let spamText = "";

const predictForm = document.getElementById("predictForm");
const spamTextID = document.getElementById("spamTextID");

spamTextID.addEventListener("change", (event) => {
  spamText = event.target.value;
});

predictForm.addEventListener("submit", (event) => {
  event.preventDefault();

  fetch(BASE_URL, { data: spamText })
    .then((data) => {
      console.log(data);
    })
    .catch((error) => {
      console.log(error);
    });
});
