const wrapper = document.querySelector('.tag-scroll-wrapper');
let isDown = false;
let startX;
let scrollLeft;

// Kéo chuột
wrapper.addEventListener('mousedown', (e) => {
  isDown = true;
  wrapper.classList.add('dragging');
  startX = e.pageX - wrapper.offsetLeft;
  scrollLeft = wrapper.scrollLeft;
});

wrapper.addEventListener('mouseleave', () => {
  isDown = false;
  wrapper.classList.remove('dragging');
});

wrapper.addEventListener('mouseup', () => {
  isDown = false;
  wrapper.classList.remove('dragging');
});

wrapper.addEventListener('mousemove', (e) => {
  if (!isDown) return;
  e.preventDefault();
  const x = e.pageX - wrapper.offsetLeft;
  const walk = (x - startX) * 1; // tốc độ cuộn
  wrapper.scrollLeft = scrollLeft - walk;
});

// Cuộn khi bấm nút
const scrollAmount = () => {
  const tag = wrapper.querySelector('.link-block-2');
  return tag ? tag.offsetWidth * 3 + 18 : 300; // cuộn 3 tag mỗi lần (18 là khoảng cách/gap nếu có)
};

document.querySelector('.scroll-button.left').addEventListener('click', () => {
  wrapper.scrollBy({ left: -scrollAmount(), behavior: 'smooth' });
});

document.querySelector('.scroll-button.right').addEventListener('click', () => {
  wrapper.scrollBy({ left: scrollAmount(), behavior: 'smooth' });
});


// Get modal elements
const loginModal = document.getElementById('loginModal');
const signupModal = document.getElementById('signupModal');

// Get buttons
const loginBtn = document.getElementById('loginBtn');
const signupBtn = document.getElementById('signupBtn');
const closeLoginModal = document.getElementById('closeLoginModal');
const closeSignupModal = document.getElementById('closeSignupModal');
const switchToSignup = document.getElementById('switchToSignup');
const switchToLogin = document.getElementById('switchToLogin');

// Open login modal
loginBtn.addEventListener('click', function() {
    loginModal.style.display = 'flex';
});

// Open signup modal
signupBtn.addEventListener('click', function() {
    signupModal.style.display = 'flex';
});

// Close login modal
closeLoginModal.addEventListener('click', function() {
    loginModal.style.display = 'none';
});

// Close signup modal
closeSignupModal.addEventListener('click', function() {
    signupModal.style.display = 'none';
});

// Switch from login to signup
switchToSignup.addEventListener('click', function() {
    loginModal.style.display = 'none';
    signupModal.style.display = 'flex';
});

// Switch from signup to login
switchToLogin.addEventListener('click', function() {
    signupModal.style.display = 'none';
    loginModal.style.display = 'flex';
});

// Close modals when clicking outside
window.addEventListener('click', function(event) {
    if (event.target == loginModal) {
        loginModal.style.display = 'none';
    }
    if (event.target == signupModal) {
        signupModal.style.display = 'none';
    }
});

// Handle login form submission
document.getElementById('loginForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    
    // Here you would typically send this data to your server
    console.log('Login attempt with:', {email, password});
    
    // For demo, just close the modal and show alert
    alert('Đăng nhập thành công!');
    loginModal.style.display = 'none';
});

// Handle signup form submission
document.getElementById('signupForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;
    
    // Simple client-side validation
    if (password !== confirmPassword) {
        alert('Mật khẩu xác nhận không khớp!');
        return;
    }
    
    // Here you would typically send this data to your server
    console.log('Signup attempt with:', {name, email, password});
    
    // For demo, just close the modal and show alert
    alert('Đăng ký thành công!');
    signupModal.style.display = 'none';
});

