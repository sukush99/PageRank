async function search() {
    let query = document.getElementById('query').value;
    let response = await fetch(`/search?query=${query}`);
    let data = await response.json();
    document.getElementById('output').innerText = data.message;
}