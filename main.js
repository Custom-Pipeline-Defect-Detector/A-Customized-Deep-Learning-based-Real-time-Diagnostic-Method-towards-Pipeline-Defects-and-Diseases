import Vue from 'vue';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import App from './App.vue';

Vue.use(ElementUI);
//设置项目中拥有size和zindex属性的默认值
Vue.use(Element, { size: 'small', zIndex: 3000 });

new Vue({
  el: '#app',
  render: h => h(App)
});
