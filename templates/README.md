### 5-7. templates/index.html

#### 5-7-1. body
```html
<!DOCTYPE html>
{% load static %}
<html lang="ko">

<head>
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>

<body>
    <h1>opt_web</h1>
    <form method="POST" onsubmit="return validateForm()">
        {% csrf_token %}
        <label for="board">Choose a board</label>
        <select name="board" id="board">
            <option value="자유">자유</option>
            <option value="비밀">비밀</option>
            <option value="신입생">신입생</option>
        </select>
        <label for="gen_num">생성할 문장의 개수</label>
        <input type="number" name="gen_num" id="gen_num">
        <label for="mxl">문장의 최대 길이</label>
        <input type="number" name="mxl" id="mxl">
        <br><br>
        <div class="button-container">
            <button type="submit">문장 생성</button>
        </div>
    </form>
    <h2>생성된 문장</h2>
    {% for text in gen_text %}
    <div>
        {{text}}
    </div>
    {% endfor %}
```
'\<head\>'섹션에서 정적 파일인 'style.css'를 불러와 페이지에 스타일을 적용한다.

'\<form\>` 태그는 사용자의 입력을 받기 위한 폼을 정의한다, 'method' 속성에 POST를 지정하여, 이 입력을 서버로 전송할 때 POST 메소드를 사용하도록 한다. 이는  **5-6-2. index**에서 조건문으로 작동할 수 있도록 하는 역할을 한다.

사용자로부터 입력받는 값은 게시판 선택 드롭다운 박스에서 "자유", "비밀", "신입생" 중 하나를 선택하고, '\<input\>' 태그를 사용하여 정수 값을 입력받는 생성할 문장의 개수와 문장의 최대 길이가 있다.

값을 입력하고 "문장 생성" 버튼을 누르면, 생성된 문장이 출력된다.

#### 5-7-2. script
```html
    <script>
        // 페이지 로드 시 localStorage에 저장된 값이 있으면 가져와서 각 입력 필드에 설정
        window.onload = function () {
            var board = localStorage.getItem('board');
            var genNum = localStorage.getItem('genNum');
            var mxl = localStorage.getItem('mxl');

            document.getElementById('board').value = board || '';
            document.getElementById('gen_num').value = genNum || '';
            document.getElementById('mxl').value = mxl || '';
        };

        function validateForm() {
            var board = document.getElementById('board').value;
            var genNum = document.getElementById('gen_num').value;
            var mxl = document.getElementById('mxl').value;

            if (board.trim() === '' || genNum.trim() === '' || mxl.trim() === '') {
                alert('모든 입력값을 입력해주세요.');
                return false; // 폼 제출 방지
            }

            // 0보다 큰 수를 입력하도록 확인
            if (genNum <= 0 || mxl <= 0) {
                alert('0보다 큰 수를 입력해주세요.');
                return false; // 폼 제출 방지
            }

            // localStorage에 값 저장
            localStorage.setItem('board', board);
            localStorage.setItem('genNum', genNum);
            localStorage.setItem('mxl', mxl);

            // 추가적인 유효성 검사 로직을 여기에 추가할 수 있습니다.
            return true; // 폼 제출 허용
        }
    </script>
</body>

</html>
```
`window.onload`는 페이지 로드 시 localStorage에 저장된 값이 있는 경우, 해당 값을 가져와서 각 입력 필드에 설정해주는 함수다. 이를 통해 사용자가 값을 지정하고 전송 버튼을 눌러도 지정한 값이 유지되게 한다.

`validateForm()` 함수는 사용자로부터 입력받은 값들의 유효성을 검사한다. 모든 필드에 대한 입력을 강제하며, 생성할 문장의 개수와 문장의 최대 길이에 대해서는 0 이상의 값이어야 한다.
